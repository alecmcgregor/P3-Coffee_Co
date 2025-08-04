import pandas as pd
import random

df = pd.read_csv('coffee.csv')
df.drop(columns=['Location.Country', 'Location.Region', 'Location.Altitude.Min', 'Location.Altitude.Max', 'Location.Altitude.Average', 'Year', 'Data.Owner', 'Data.Type.Species', 'Data.Type.Variety', 'Data.Type.Processing method', 'Data.Production.Bag weight', 'Data.Production.Number of bags', 'Data.Scores.Total', 'Data.Color'], inplace = True)
rows = []
for i in range(99010):
    aroma = round(random.uniform(0,10), 2)
    flavor = round(random.uniform(0, 10), 2)
    aftertaste = round(random.uniform(0, 10), 2)
    acidity = round(random.uniform(0, 10), 2)
    body = round(random.uniform(0, 10), 2)
    balance = round(random.uniform(0, 10), 2)
    uniformity = round(random.uniform(0, 10), 2)
    sweetness = round(random.uniform(0, 10), 2)
    moisture = round(random.uniform(0, 0.30), 2)
    rows.append({
        "Data.Scores.Aroma" : aroma,
        "Data.Scores.Flavor" : flavor,
        "Data.Scores.Aftertaste" : aftertaste,
        "Data.Scores.Acidity" : acidity,
        "Data.Scores.Body" : body,
        "Data.Scores.Balance" : balance,
        "Data.Scores.Uniformity" : uniformity,
        "Data.Scores.Sweetness" : sweetness,
        "Data.Scores.Moisture" : moisture,
    })

new_df = pd.DataFrame(rows)
final_df = pd.concat([df, new_df], ignore_index = True)
final_df.to_csv("generated_coffee.csv", index=False, na_rep="nan")
