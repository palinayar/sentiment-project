from RobertaModel import RobertaModel
from VaderModel import VaderModel
from process_files_parallel import process_folder
import gc

def main():
    try:
        # Initialize models
        roberta_model = RobertaModel()
        if not roberta_model.Initialized:
            return "Unable to initialize ROBERTA model"
            
        vader_model = VaderModel()
        if not vader_model.Initialized:
            return "Unable to initialize VADER model"

        # For parallel processing:
        process_folder(roberta_model, vader_model)
    finally:
        gc.collect()
        print("Resources have been released. Script completed successfully.")

if __name__ == '__main__':
    main()
