"""
Central Management Server (RailVision)
--------------------------------------
Handles API routing, global state synchronization, historical logging, 
predictive passenger distribution, and Dynamic Loco Reversal.
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import base64
import cv2
import numpy as np
import csv
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

print("[SYSTEM] Loading YOLOv8 Segmentation Model for Validation Panel...")
model = YOLO("yolov8s-seg.pt") 

TRAIN_DB = {}
ALL_STATIONS = set()
UTS_DB = {}

GLOBAL_STATE = {
    "active_train": None,
    "station_index": 0
}

train_state_coaches = {
    "GEN-1": {"yolo_base": 0, "image": None},
    "GEN-2": {"yolo_base": 0, "image": None},
    "GEN-3": {"yolo_base": 0, "image": None},
    "GEN-4": {"yolo_base": 0, "image": None},
    "SLR-1": {"yolo_base": 0, "image": None},
    "SLR-2": {"yolo_base": 0, "image": None},
    "SLR-M": {"yolo_base": 0, "image": None}
}

STATION_NAMES = { "SC": "Secunderabad", "GNT": "Guntur", "BZA": "Vijayawada", "VSKP": "Visakhapatnam" }
def format_station(code): return STATION_NAMES.get(code, code)

def load_train_database():
    try:
        with open('train_database.csv', mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                composition = [{"id": "LOCO", "type": "LOCO", "flags": []}]
                gen_count, rsrv_count = 1, 1
                
                if "1-Front" in row.get('Female_Coach_Pos', ''): composition.append({"id": "SLR-1", "type": "SLR", "flags": ["disabled"]})
                for _ in range(int(row.get('Gen_Front', 0))):
                    composition.append({"id": f"GEN-{gen_count}", "type": "GEN", "flags": []}); gen_count += 1
                for _ in range(int(row.get('Reserved_Coaches', 0))):
                    composition.append({"id": f"RSRV-{rsrv_count}", "type": "RSRV", "flags": []}); rsrv_count += 1
                if "1-Middle" in row.get('Female_Coach_Pos', ''): composition.append({"id": "SLR-M", "type": "SLR", "flags": []})
                for _ in range(int(row.get('Gen_Back', 0))):
                    composition.append({"id": f"GEN-{gen_count}", "type": "GEN", "flags": []}); gen_count += 1
                if "1-Back" in row.get('Female_Coach_Pos', ''): composition.append({"id": "SLR-2", "type": "SLR", "flags": ["female"]})
                    
                stations_data = []
                for idx, st in enumerate(row['Halts'].split('|')):
                    fname = format_station(st)
                    ALL_STATIONS.add(fname)
                    stations_data.append({"name": fname, "code": st, "arr": "--:--", "dep": "--:--"})
                
                # --- UPDATED: Sets BOTH BZA and VSKP as reversal stations ---
                reversals = row.get('Reversal_Stations', 'BZA|VSKP').split('|')
                
                t_no = row['Train_No']
                TRAIN_DB[t_no] = {
                    "number": t_no, "name": row['Train_Name'], "type": row['Train_Type'],
                    "route_start": format_station(row['Origin']), "route_end": format_station(row['Destination']),
                    "stations": stations_data, "composition": composition,
                    "reversals": [r for r in reversals if r]
                }
                UTS_DB[t_no] = {st['name'].split(' (')[0]: {'boarding': 0, 'departing': 0} for st in stations_data}
    except Exception as e: print(f"[ERROR] DB Load Failed: {e}")

load_train_database()

PASSENGER_DENSITY_IMPACT = 0.83 

def distribute_passengers(total_passengers, base_densities, is_boarding=True):
    distributed_changes = {k: 0 for k in base_densities.keys()}
    if total_passengers <= 0: return distributed_changes

    if is_boarding: weights = {k: max(0, 100 - v)**2 for k, v in base_densities.items()}
    else: weights = {k: v**2 for k, v in base_densities.items()}
        
    total_weight = sum(weights.values())
    if total_weight == 0:
        weights = {k: 1 for k in base_densities.keys()}
        total_weight = len(base_densities)

    for k in distributed_changes.keys():
        passengers_in_coach = total_passengers * (weights[k] / total_weight)
        distributed_changes[k] = passengers_in_coach * PASSENGER_DENSITY_IMPACT

    return distributed_changes

@app.route('/')
def passenger_dashboard(): return render_template('index.html')

@app.route('/admin')
def admin_dashboard(): return render_template('admin.html')

@app.route('/uts')
def uts_dashboard(): return render_template('uts.html')

@app.route('/api/stations', methods=['GET'])
def get_stations(): return jsonify(sorted(list(ALL_STATIONS)))

@app.route('/api/train_list', methods=['GET'])
def get_train_list(): return jsonify([{"number": t["number"], "name": t["name"]} for t in TRAIN_DB.values()])

@app.route('/api/get_train/<train_no>', methods=['GET'])
def get_train(train_no):
    if train_no in TRAIN_DB: return jsonify(TRAIN_DB[train_no])
    return jsonify({"error": "Train not found"}), 404

@app.route('/api/search_route', methods=['GET'])
def search_route():
    src = request.args.get('src', '').lower()
    dest = request.args.get('dest', '').lower()
    results = []
    for t_no, t_data in TRAIN_DB.items():
        st_names = [s['name'].lower() for s in t_data['stations']]
        src_match = next((s for s in st_names if src in s), None)
        dest_match = next((s for s in st_names if dest in s), None)
        if src_match and dest_match and st_names.index(src_match) < st_names.index(dest_match):
            results.append({"number": t_no, "name": t_data['name'], "type": t_data['type'], "departure": t_data['stations'][st_names.index(src_match)]['dep']})
    return jsonify(results)

@app.route('/api/book_uts', methods=['POST'])
def book_uts():
    data = request.json
    t_no = data['train_no']
    src = data['source']
    dest = data['destination']
    count = data['passengers']
    if t_no in UTS_DB:
        if src in UTS_DB[t_no]: UTS_DB[t_no][src]['boarding'] += count
        if dest in UTS_DB[t_no]: UTS_DB[t_no][dest]['departing'] += count
    return jsonify({"status": "success"})

@app.route('/api/reset_uts', methods=['POST'])
def reset_uts():
    for t_no in UTS_DB:
        for st in UTS_DB[t_no]:
            UTS_DB[t_no][st]['boarding'] = 0
            UTS_DB[t_no][st]['departing'] = 0
    return jsonify({"status": "success"})

@app.route('/api/admin/set_train', methods=['POST'])
def set_train():
    GLOBAL_STATE['active_train'] = request.json['train_no']
    GLOBAL_STATE['station_index'] = 0
    return jsonify({"status": "success"})

@app.route('/api/admin/depart', methods=['POST'])
def depart():
    if not GLOBAL_STATE['active_train']: return jsonify({"error": "No train active"})
    t_no = GLOBAL_STATE['active_train']
    train_info = TRAIN_DB[t_no]
    curr_idx = GLOBAL_STATE['station_index']
    
    if curr_idx >= len(train_info['stations']) - 1: return jsonify({"error": "End of journey reached"})

    curr_st = train_info['stations'][curr_idx]['name'].split(' (')[0]
    next_st = train_info['stations'][curr_idx + 1]['name'].split(' (')[0]

    total_crowd = sum(c['yolo_base'] for c in train_state_coaches.values())
    avg_val = round(total_crowd / max(1, len(train_state_coaches)), 2)
    crowd_cat = "Critical" if avg_val > 85 else "High" if avg_val > 60 else "Medium" if avg_val > 30 else "Low"
    
    row = [datetime.now().strftime("%d-%m-%Y"), datetime.now().strftime("%A"), t_no, train_info['route_start'], train_info['route_end'], curr_st, next_st, avg_val, crowd_cat]
    file_exists = os.path.isfile('historical_crowd_data.csv')
    with open('historical_crowd_data.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists: writer.writerow(['Date', 'Day', 'Train No', 'Route Start', 'Route End', 'Current Station', 'Next Station', 'Average Crowd', 'Crowd Category'])
        writer.writerow(row)

    GLOBAL_STATE['station_index'] += 1
    
    if GLOBAL_STATE['station_index'] >= len(train_info['stations']) - 1:
        if t_no in UTS_DB:
            for st in UTS_DB[t_no]:
                UTS_DB[t_no][st]['boarding'] = 0
                UTS_DB[t_no][st]['departing'] = 0

    return jsonify({"status": "success", "new_index": GLOBAL_STATE['station_index']})

@app.route('/api/update_crowd', methods=['POST'])
def update_crowd():
    data = request.json
    coach_id = data.get('coach_id')
    if coach_id in train_state_coaches:
        train_state_coaches[coach_id]['yolo_base'] = data.get('crowd_percent')
        current_time = int(datetime.now().timestamp())
        train_state_coaches[coach_id]['image'] = f"/static/captures/{coach_id}_live.jpg?t={current_time}"
    return jsonify({"status": "success"})

@app.route('/api/get_live_status', methods=['GET'])
def get_live_status(): 
    t_no = GLOBAL_STATE['active_train']
    if not t_no: return jsonify({"active_train": None})

    train_info = TRAIN_DB[t_no]
    curr_idx = GLOBAL_STATE['station_index']
    total_stations = len(train_info['stations'])
    
    is_final_station = (curr_idx >= total_stations - 1)
    is_penultimate_station = (curr_idx == total_stations - 2)

    next_st = train_info['stations'][curr_idx + 1]['name'].split(' (')[0] if not is_final_station else None

    # --- NEW: LOCO REVERSAL LOGIC ---
    reversal_count = 0
    # Check if we have arrived at or passed any reversal stations
    for i in range(curr_idx + 1):
        if i < len(train_info['stations']):
            st_code = train_info['stations'][i]['code']
            if st_code in train_info.get('reversals', []):
                reversal_count += 1
                
    active_comp = list(train_info['composition'])
    # If the train has reversed an odd number of times, flip the coaches
    is_reversed = False
    if reversal_count % 2 == 1:
        is_reversed = True
        loco = active_comp.pop(0) # Remove engine
        active_comp.reverse()     # Flip coaches
        active_comp.insert(0, loco) # Put engine back on the new front

    next_departing = UTS_DB[t_no][next_st]['departing'] if next_st and next_st in UTS_DB[t_no] else 0
    next_boarding = UTS_DB[t_no][next_st]['boarding'] if next_st and next_st in UTS_DB[t_no] else 0

    live_coaches = {}
    current_live_densities = {}
    
    for coach_id, cdata in train_state_coaches.items():
        current_live_densities[coach_id] = cdata['yolo_base']

    next_dep_dist = distribute_passengers(next_departing, current_live_densities, is_boarding=False)

    post_halt_densities = {}
    for coach_id, live_crowd in current_live_densities.items():
        post_halt_densities[coach_id] = max(0, live_crowd - next_dep_dist[coach_id])

    next_brd_dist = distribute_passengers(next_boarding, post_halt_densities, is_boarding=True)

    for coach_id, live_crowd in current_live_densities.items():
        if is_final_station or is_penultimate_station: expected = 0 
        else: expected = post_halt_densities[coach_id] + next_brd_dist[coach_id]

        live_coaches[coach_id] = {
            "crowd": live_crowd, 
            "expected": max(0, min(int(expected), 100)),
            "image": train_state_coaches[coach_id]['image']
        }

    return jsonify({
        "active_train": t_no, 
        "station_index": curr_idx, 
        "train_data": train_info, 
        "coaches": live_coaches,
        "active_composition": active_comp, # Backend sends the flipped array!
        "is_reversed": is_reversed
    })

@app.route('/api/demo_process', methods=['POST'])
def demo_process():
    try:
        data = request.json
        encoded_data = data.get('image').split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        
        results = model.predict(img, classes=[0], conf=0.25, verbose=False)
        if len(results[0].boxes) == 0 or results[0].masks is None:
            return jsonify({"status": "success", "density": 0, "processed_image": data.get('image')})

        masks = results[0].masks.data.cpu().numpy()
        combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
        combined_mask = cv2.resize(combined_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        ceiling_cutoff = int(h * 0.30)
        roi_mask = combined_mask[ceiling_cutoff:h, 0:w]
        
        occupied_human_pixels = cv2.countNonZero(roi_mask)
        total_roi_pixels = w * (h - ceiling_cutoff)
        
        pixel_fill_ratio = occupied_human_pixels / total_roi_pixels
        MAX_FILL_RATIO = 0.50 
        
        density = int((pixel_fill_ratio / MAX_FILL_RATIO) * 100)
        density = min(max(density, 0), 100) 

        visual_img = img.copy()
        color_overlay = np.zeros_like(visual_img)
        mask_color = [0, 0, 255] if density >= 85 else [0, 255, 0] 
        color_overlay[:, :] = mask_color
        
        colored_mask = cv2.bitwise_and(color_overlay, color_overlay, mask=combined_mask)
        cv2.addWeighted(colored_mask, 0.6, visual_img, 1.0, 0, visual_img)
        cv2.line(visual_img, (0, ceiling_cutoff), (w, ceiling_cutoff), (0, 255, 255), 2)
        
        status_color = (0, 0, 255) if density >= 85 else ((0, 255, 255) if density >= 50 else (0, 255, 0))
        cv2.putText(visual_img, f"Human Pixel Area: {int(pixel_fill_ratio*100)}%", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(visual_img, f"Calculated Density: {density}%", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
        
        _, buffer = cv2.imencode('.jpg', visual_img)
        processed_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({"status": "success", "density": density, "processed_image": processed_b64})
    except Exception as e: return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    if not os.path.exists('static/captures'): os.makedirs('static/captures')
    app.run(debug=True, host='0.0.0.0', port=5000)