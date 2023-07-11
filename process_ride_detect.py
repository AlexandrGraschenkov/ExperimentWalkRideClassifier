import os
from joblib import dump, load

class WalkDetect:
    def __init__(self, model_path):
        self.model = load(model_path)

    def prepare_data(file):
        import pandas as pd
        import numpy as np
        data = pd.read_csv(file)
        fields = ["rrate_x","rrate_y","rrate_z",
                "gravity_x","gravity_y","gravity_z",
                "accel_x","accel_y","accel_z"]
        std_values = [np.std(data[f]) for f in fields] 
        return std_values
    
    def _check_is_ride_by_gps_speed(self, gps_file):
        import pandas as pd
        data = pd.read_csv(gps_file)
        speed = data["speed"]
        if len(speed) == 0: 
            return False
        
        max_speed_kmh = max(speed) * 3.6
        return max_speed_kmh > 15
    
    def _check_is_walk_by_motion_data(self, motion_file):
        import pandas as pd
        import numpy as np
        data = pd.read_csv(motion_file)
        std_val = np.std(data["rrate_z"])
        # обычно у пользователя в руках тряска поворачивает сильно телефон по оси Z
        return std_val > 0.12 

    def check_is_walk(self, motion_file, gps_file = None):
        if gps_file and self._check_is_ride_by_gps_speed(gps_file):
            return False # 100% ride
        
        if self._check_is_walk_by_motion_data(motion_file):
            return True
        
        values = WalkDetect.prepare_data(motion_file)
        res = self.model.predict([values])[0]
        return res == "walk"
    
if __name__ == '__main__':
    d = WalkDetect("model.joblib")

    root_data = "/Users/alex/Downloads/data/test"
    for f in sorted(os.listdir(root_data)):
        if not f.endswith(".csv"): continue
        ff = os.path.join(root_data, f)
        res = d.check_is_walk(ff)
        print(ff, "walk" if res else "-")

    # res = d.check_is_walk("/Users/alex/Downloads/2023-07-11_10-37-09/motion.csv", 
    #                       gps_file="/Users/alex/Downloads/2023-07-11_10-37-09/gps.csv")
    # print("Last", "walk" if res else "-")
    # d.check_is_walk("")