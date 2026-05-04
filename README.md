🚂 RailVision: Smart Coach Management SystemRailVision is an AI-powered predictive intelligence platform designed to eliminate chaotic boarding and uneven passenger distribution in unreserved railway coaches. By bridging physical camera telemetry with digital ticketing data, it transforms passive train monitoring into an active, real-time crowd distribution engine. 


✨ Why RailVision?In the current railway landscape, passengers often face "boarding guesswork". One coach may be packed beyond capacity while the adjacent one remains nearly empty, simply because passengers lack real-time visibility.  RailVision solves this by providing:Real-Time Crowd Visibility: Exact occupancy data for every coach.  Predictive Intelligence: Guidance to the "Optimal Coach" before the train even arrives.  Safety & Equity: Eliminating dangerous station rushes and ensuring a balanced passenger load.

🛠️ The Tech StackRailVision utilizes a sophisticated "Edge-to-Cloud" architecture to ensure low-latency processing and high accuracy.  
🧠 Edge AI & Computer VisionUnlike traditional systems that use simple bounding boxes (which fail during overlaps), RailVision employs YOLOv8s Instance Segmentation.  Perspective-Weighted Footprint Mapping: Extracts precise 2D pixel masks for each person to calculate actual floor occupancy rather than just "counting heads".  Occlusion Handling: Effectively manages crowded environments where passengers may overlap in the camera's line of sight.  
☁️ Cloud & IntelligenceStateful UTS Engine: Merges live camera feeds with Cloud-based ticketing (UTS) data.  Inverse Proportional Boarding Algorithm: A custom algorithm that calculates weights based on current density to distribute incoming passengers scientifically.  Transit Event Handling: Native support for complex railway scenarios like Locomotive Reversals.



🚀 Key FeaturesLive Multi-lingual Dashboard: A web-based interface providing real-time telemetry for passengers and administrators.  Edge Device Optimization: Designed to run on low-cost hardware like Raspberry Pi 4 or Jetson Nano.  High Accuracy: Outperforms traditional object detection in accuracy, noise reduction, and occlusion handling.  Accessibility: Includes a mobile app and web features like the Web Speech API for audio status updates. 


📈 Real-World ApplicationsRailVision is built for scale and supports various transit systems:  EMU/MEMU (Local/Mainline Electric Multiple Units)   Metro Trains   Express Trains (General/Unreserved Coaches)  


🔮 Future ScopeWe are looking to expand RailVision's capabilities through:  IRCTC Integration: Native API embedding into official railway apps.Night Vision: Thermal-trained models for overnight routes.ML Forecasting: Predictive crowd patterns based on historical station data.

Project Developed By: > Y Hemanth Kumar, K Rohith, DV Ram Charan, and B Mothilal NaikDepartment of Computer Science & Engineering, RGUKT Nuzvid  
