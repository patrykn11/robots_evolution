import os
import json
import pickle
import zipfile
import glob


class SavingMixin:
    def save_history(self, history):
        if not hasattr(self, 'save_path'):
            return

        with open(os.path.join(self.save_path, "history.json"), "w") as f:
            json.dump(history, f, indent=4)

    def save_robot(self, generation, robot):
        if not hasattr(self, 'save_path') or robot is None:
            return

        with open(
            os.path.join(self.save_path, f"gen_{generation:03d}.pkl"),
            "wb"
        ) as f:
            pickle.dump(robot, f)

    def save_species_champions(self, generation, species_list):
        """Save best individual from each species to results/<exp>/species/"""
        if not hasattr(self, 'save_path'):
            return

        species_dir = os.path.join(self.save_path, "species")
        os.makedirs(species_dir, exist_ok=True)

        for i, s in enumerate(species_list):
            if len(s.members) == 0:
                continue

            champion = max(s.members, key=lambda ind: ind.fitness)
            filename = f"species_{i:02d}_gen_{generation:03d}.pkl"

            with open(os.path.join(species_dir, filename), "wb") as f:
                pickle.dump(champion, f)

    def zip_results(self):
        if not hasattr(self, 'save_path'):
            return

        zip_filename = os.path.join(self.save_path, "evolution_data.zip")
        pkl_files = glob.glob(os.path.join(self.save_path, "*.pkl"))

        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in pkl_files:
                zipf.write(file, os.path.basename(file))

        for file in pkl_files:
            os.remove(file)