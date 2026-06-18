from datasets import load_dataset

dataset = load_dataset("merve/poetry")
df = dataset["train"].to_pandas()
df.to_csv("poetry.csv", index=False)
print(df.head())