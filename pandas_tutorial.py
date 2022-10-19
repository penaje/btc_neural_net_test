import re
import pandas as pd

pd.set_option("display.max_columns", None, 'display.max_rows', 0)
pd.set_option('display.width', 200)

df = pd.read_csv('pokemon_data.csv')

print(df.tail(5))

## Read Headers

print(df.columns)

## Read each Column

print(df["Name"])
print(df[['Name', 'Type 1', 'HP']])

## Read a Row

print(df.iloc[2])

## Read a Specific Location

print(df.iloc[2, 3])

## Iterate through each Row in DataFrame
for index, row in df.iterrows():
    print(index, row)

for index, row in df.iterrows():
    print(index, row['Name'])

## Print only "Fire type Pokemon"
print(df.loc[df['Type 1'] == 'Fire'])

## Show Stats
print(df.describe())

## Sort w/multiple columns and different types

print(df.sort_values(['Type 1', 'HP'], ascending=[1, 0]))

## Making changes to the data

# Create Total Column
df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']

print(df.head(5))

# Remove Total Column
df = df.drop(columns='Total')

print(df.head(5))

# Recreate Total Column
df['Total'] = df.iloc[:, 4:10].sum(axis=1)

print(df.head())

cols = list(df.columns.values)

# Reorder the columns to move Totals
df = df[cols[0:4] + [cols[-1]] + cols[4:12]]

print(df.head(5))

# # Create new modified CSV
df.to_csv('modified_pokemon.csv', index=False)

## Filtering Data

new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison') & (df['HP'] > 70)]

print(new_df)

## Reset the Index

newer_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison') & (df['HP'] > 70)].reset_index(drop=True)

print(newer_df)

## Reset Index In Place

new_df.reset_index(drop=True, inplace=True)

print(new_df)

## Remove 'Mega" Pokemon

print("\nMEGA\n")
# Get all the rows that contain 'Mega'
print(df.loc[df['Name'].str.contains('Mega')])

print("\nNOT MEGA\n")
# Get all the rows that do NOT contain 'Mega'
print(df.loc[~df['Name'].str.contains('Mega')])

## REGEX

print("\nREGEX\n")
print(df.loc[df['Type 1'].str.contains('fire|grass', flags=re.IGNORECASE, regex=True)])

## Conditional Changes

df.loc[df['Type 1'] == 'Fire', 'Type 1'] = 'Flamer'

print(df.head())

## Grouping/Aggregate Stats

print(df.groupby(['Type 1']).mean().sort_values('Defense', ascending=False))

print(df.groupby(['Type 1']).mean().sort_values('Attack', ascending=False))
