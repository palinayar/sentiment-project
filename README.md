# sentiment-project

### Run analysis

1. Create environment:

```
python -m venv env
source env/bin/activate   # For Linux/Mac
.\env\Scripts\activate    # For Windows
```

2. Run setup

```
python setup_environment.py
```

3. Run sentiment analysis

```
python main.py
```

See output csv files in **output_uniq** folder

To limit amount of data that will be processed, set the *ROWS_TO_PROCESS* variable in process_csv_file.py accordingly.
