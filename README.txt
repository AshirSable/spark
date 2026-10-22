MLOps ICE 2 - SOLUTIONS (one folder, three parts)

1_Linux_Q11-15/
    linux_solutions.txt         every sub-question (a, b, c ...) with the command and expected output

2_YAML_RandomForest_Q1-5/       Random Forest + saved .pkl, in the style of your YAML practical
    Q1_Laptop_Price ... Q5_Employee_Salary
    each folder: config.yaml, train.py, your Excel data file, expected_output.txt, README.txt
    run:  pip install pyyaml scikit-learn joblib pandas openpyxl
          python train.py          (inside the question's folder)

3_Feature_Engineering_Q6-10/    Google Colab style, same steps as the telecom tower practical
    Q6 ... Q10
    each folder: solution.py (paste into ONE Colab cell), your Excel data file, expected_output.txt, README.txt

NOTES
- Everything was run, and the outputs saved, on your real Excel files.
- Q3 is cut off in the question paper after task 2. I assumed the same tasks as the others and a made-up test student. Check your paper.
- Q6 (customer churn): no data file was in your zip. Its folder has SAMPLE data with assumed columns. Change the names to match the real file.
- Q7 to Q10 thresholds (for example Age > 60) are my choices from the data. Be ready to explain them.
- Your real Q1-5 numbers can differ slightly with another scikit-learn version.
