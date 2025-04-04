from datasets import load_dataset
import ast
# import pandas as pd

# describe load_dataset function
print(dir(load_dataset))
help(load_dataset)


# dowload the osunlp/TravelPlanner dataset
dataset = load_dataset("osunlp/TravelPlanner", "test")

# # and print the first 5 rows
# print(dataset['test'][:1])

# transform the dataset into a pandas dataframe
df = dataset['test'].to_pandas()

# # print the first 5 rows of the dataframe
# print(df.head())


# Función para convertir el contenido de 'reference_information'
def parse_reference_info(ref):
    # Se asume que cada valor es una lista con un único string que contiene la representación del JSON
    if isinstance(ref, list) and len(ref) == 1 and isinstance(ref[0], str):
        try:
            # Usamos ast.literal_eval para evaluar el string como un literal de Python
            parsed = ast.literal_eval(ref[0])
            return parsed
        except Exception as e:
            print(f"Error al parsear: {ref}. Error: {e}")
            return None
    else:
        try:
            return ast.literal_eval(ref)
        except Exception as e:
            print(f"Error al parsear: {ref}. Error: {e}")
            return None

# Aplicar la función a la columna 'reference_information'
df['reference_information_parsed'] = df['reference_information'].apply(parse_reference_info)


# df['reference_information_parsed'][0]
# len(df['reference_information_parsed'][0])


# save the dataframe to a pickle file
path = r'data/TravelPlanner.pkl'
df.to_pickle(path)

