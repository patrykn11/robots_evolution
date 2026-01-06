import os
import json
import pickle
import zipfile
import glob

class SavingMixin:
    def save_history(self, history):
        """Saves the history dictionary to a JSON file."""
        if not hasattr(self, 'save_path'):
            return
            
        with open(os.path.join(self.save_path, "history.json"), "w") as f:
            json.dump(history, f, indent=4)

    def save_robot(self, generation, robot):
        """Saves the robot object to a pickle file."""
        if not hasattr(self, 'save_path') or robot is None:
            return

        with open(os.path.join(self.save_path, f"gen_{generation:03d}.pkl"), "wb") as f:
            pickle.dump(robot, f)

    def zip_results(self):
        """Zips all .pkl files in the save path into a single zip file.
           Removes the original .pkl files after zipping.
        """
        if not hasattr(self, 'save_path'):
            return
            
        zip_filename = os.path.join(self.save_path, "evolution_data.zip")
        pkl_files = glob.glob(os.path.join(self.save_path, "*.pkl"))
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in pkl_files:
                zipf.write(file, os.path.basename(file))
                
        for file in pkl_files:
            os.remove(file)
