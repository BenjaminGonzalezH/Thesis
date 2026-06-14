"""
extraer_genes_matrices.py
--------------------------
Extrae todos los genes únicos (ENSG) presentes en una o varias matrices de
distancia/correlación CSV y genera un subconjunto filtrado del archivo de
mapeo ENSG → nombre de gen.

Estructura esperada de las matrices:
  - Primera fila  : cabecera con IDs de gen (col 0 puede estar vacía o ser un índice)
  - Primera columna: IDs de gen (mismos que la cabecera, matriz simétrica)
  Los genes se recogen de AMBOS sitios para no depender de la orientación.

Uso:
    python3 extraer_genes_matrices.py \\
        --mapeo   archivo.csv \\
        --dir     /ruta/a/matrices \\
        --patron  "*.csv" \\
        --out     genes_filtrados.csv

Argumentos opcionales:
    --mapeo-ensg  Columna ENSG en el mapeo  (default: 'incoming')
    --mapeo-name  Columna nombre en el mapeo (default: 'name')
    --excluir     Archivos a ignorar del directorio, separados por comas
                  (ej: "archivo.csv,otro.csv")
    --solo-ensg   Si se indica, el CSV de salida solo contiene las dos columnas
                  del mapeo (sin columna de status de validación GAF)
"""

import argparse
import glob
import os
import csv
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mapeo",      required=True,
                        help="CSV de mapeo ENSG → nombre (ej: archivo.csv)")
    parser.add_argument("--dir",        default=".",
                        help="Directorio donde buscar las matrices (default: '.')")
    parser.add_argument("--patron",     default="*.csv",
                        help="Patrón glob para los archivos de matriz (default: '*.csv')")
    parser.add_argument("--out",        default="genes_filtrados.csv",
                        help="Archivo de salida con el mapeo filtrado")
    parser.add_argument("--mapeo-ensg", default="incoming",
                        help="Columna ENSG en el archivo de mapeo (default: 'incoming')")
    parser.add_argument("--mapeo-name", default="name",
                        help="Columna nombre en el archivo de mapeo (default: 'name')")
    parser.add_argument("--excluir",    default="",
                        help="Nombres de archivo a excluir, separados por comas")
    return parser.parse_args()


def is_ensg(value: str) -> bool:
    """Comprueba si un valor parece un ID Ensembl humano."""
    v = value.strip()
    return v.startswith("ENSG") and len(v) > 4


def extraer_genes_de_matriz(path: str) -> set:
    """
    Lee un CSV de matriz de distancia/correlación y devuelve el conjunto
    de IDs Ensembl encontrados en la cabecera y en la primera columna.
    
    Estrategia liviana: solo lee la primera fila (cabecera) y la primera
    columna de cada fila, sin cargar toda la matriz en memoria.
    """
    genes = set()
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for i, row in enumerate(reader):
                if not row:
                    continue
                if i == 0:
                    # Cabecera: todos los campos que parezcan ENSG
                    for cell in row:
                        if is_ensg(cell):
                            genes.add(cell.strip())
                else:
                    # Primera columna de cada fila de datos
                    if is_ensg(row[0]):
                        genes.add(row[0].strip())
    except Exception as e:
        print(f"  [AVISO] No se pudo leer {path}: {e}", file=sys.stderr)
    return genes


def main():
    args = parse_args()

    excluidos = {f.strip() for f in args.excluir.split(",") if f.strip()}
    mapeo_basename = os.path.basename(args.mapeo)

    # ── 1. Descubrir archivos de matriz ───────────────────────────────────────
    patron_completo = os.path.join(args.dir, args.patron)
    archivos = sorted(glob.glob(patron_completo))

    # Excluir el propio archivo de mapeo y los indicados por el usuario
    archivos = [
        f for f in archivos
        if os.path.basename(f) not in excluidos
        and os.path.basename(f) != mapeo_basename
    ]

    if not archivos:
        print(f"No se encontraron archivos con el patrón '{patron_completo}'.")
        sys.exit(1)

    print(f"[1/3] Matrices encontradas: {len(archivos)}")
    for f in archivos:
        print(f"      {os.path.basename(f)}")

    # ── 2. Extraer genes únicos de todas las matrices ─────────────────────────
    print("\n[2/3] Extrayendo genes de las matrices...")
    genes_requeridos = set()
    for path in archivos:
        antes = len(genes_requeridos)
        nuevos = extraer_genes_de_matriz(path)
        genes_requeridos |= nuevos
        print(f"      {os.path.basename(path):40s}  +{len(nuevos):>6,}  (total: {len(genes_requeridos):,})")

    print(f"\n  → {len(genes_requeridos):,} genes únicos en total.")

    # ── 3. Filtrar el archivo de mapeo ────────────────────────────────────────
    print(f"\n[3/3] Filtrando mapeo: {args.mapeo}")
    encontrados = 0
    no_encontrados = genes_requeridos.copy()

    # Verificar columnas antes de escribir nada
    with open(args.mapeo, newline="", encoding="utf-8") as fin:
        reader_check = csv.DictReader(fin)
        if args.mapeo_ensg not in reader_check.fieldnames:
            print(f"ERROR: columna '{args.mapeo_ensg}' no encontrada en {args.mapeo}.")
            print(f"  Columnas disponibles: {reader_check.fieldnames}")
            sys.exit(1)

    with open(args.mapeo, newline="", encoding="utf-8") as fin, \
         open(args.out,   "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            ensg = row[args.mapeo_ensg].strip()
            if ensg in genes_requeridos:
                writer.writerow(row)
                encontrados += 1
                no_encontrados.discard(ensg)

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f"\n── Resumen ─────────────────────────────────────────")
    print(f"  Genes requeridos (en matrices)  : {len(genes_requeridos):>8,}")
    print(f"  Genes hallados en el mapeo      : {encontrados:>8,}")
    print(f"  Genes sin entrada en el mapeo   : {len(no_encontrados):>8,}")
    print(f"\n  Archivo de salida: {args.out}")

    if no_encontrados:
        sin_mapeo_path = args.out.replace(".csv", "_sin_mapeo.txt")
        with open(sin_mapeo_path, "w", encoding="utf-8") as f:
            for g in sorted(no_encontrados):
                f.write(g + "\n")
        print(f"  Genes sin mapeo guardados en  : {sin_mapeo_path}")


if __name__ == "__main__":
    main()