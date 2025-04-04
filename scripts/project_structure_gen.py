import os

def imprimir_estructura(directorio, nivel=0):
    try:
        with os.scandir(directorio) as it:
            for entrada in it:
                # Omitir el directorio .venv
                if entrada.name in ['pipvenv', '.venv', 'package', '.git', 'project_structure_gen.py'] and entrada.is_dir():
                    continue
                print('    ' * nivel + '|-- ' + entrada.name)
                if entrada.is_dir():
                    imprimir_estructura(entrada.path, nivel + 1)
    except PermissionError:
        print('    ' * nivel + '|-- [Acceso denegado]')
    except FileNotFoundError:
        print('    ' * nivel + '|-- [Directorio no encontrado]')
    except Exception as e:
        print(f'    {"    " * nivel}|-- [Error: {e}]')

# Ruta del directorio raíz de tu proyecto
ruta_proyecto = r'C:\Users\MANUEL ALEJANDRO\Documentos\FiftWall-test-case'

# Llamar a la función para imprimir la estructura
imprimir_estructura(ruta_proyecto)
