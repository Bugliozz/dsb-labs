# Exercise 1

This chapter groups the introductory plotting, EDA, and Docker exercises.

## Local runner

Install chapter dependencies:

```bash
python -m pip install -r exercise_1/requirements.txt
```

Run the chapter-local launcher:

```bash
python exercise_1/main.py
```

Run one or more specific steps:

```bash
python exercise_1/main.py --exercise 1_2
python exercise_1/main.py --exercise 1_3 1_5
python exercise_1/main.py --exercise all
```

## Steps

- `1_2`: save the multi-panel matplotlib layout into `exercise_1/step_2/images/`
- `1_3`: save the composed seaborn figure into `exercise_1/step_3/images/`
- `1_4`: run EDA on the liver dataset in `exercise_1/step_4/datasets/`
- `1_5`: recreate the ETF correlation visuals in `exercise_1/step_5/images/`
- `1_6`: validate the committed Docker + DokuWiki scaffold and print next commands
- `1_7`: validate the committed Flask + Docker scaffold and print next commands

Use the root `main.py` only if you want the top-level launcher that dispatches to all chapters.
