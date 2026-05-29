"""
automate_Nama-siswa.py
=======================
Script otomatisasi preprocessing untuk dataset Titanic.
Konversi dari notebook eksperimen ke pipeline preprocessing otomatis.

Penggunaan:
    python automate_Nama-siswa.py [--input titanic_raw.csv] [--output titanic_preprocessing.csv]

Output:
    - titanic_preprocessing.csv (dataset siap training)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import argparse
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_data(input_path: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV atau URL.
    
    Args:
        input_path: Path ke file CSV atau URL dataset
        
    Returns:
        DataFrame berisi dataset mentah
    """
    logger.info(f"Memuat dataset dari: {input_path}")
    
    if input_path.startswith("http"):
        df = pd.read_csv(input_path)
    else:
        if not os.path.exists(input_path):
            # Fallback: download dari URL default jika file tidak ditemukan
            logger.warning(f"File {input_path} tidak ditemukan. Mencoba download dari URL default...")
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            df = pd.read_csv(url)
        else:
            df = pd.read_csv(input_path)
    
    logger.info(f"Dataset berhasil dimuat. Shape: {df.shape}")
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus kolom yang tidak relevan untuk model machine learning.
    
    Args:
        df: DataFrame input
        
    Returns:
        DataFrame tanpa kolom tidak relevan
    """
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols)
    logger.info(f"Kolom dihapus: {existing_cols}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat fitur baru dari fitur yang ada (feature engineering).
    
    Args:
        df: DataFrame input
        
    Returns:
        DataFrame dengan fitur baru
    """
    logger.info("Melakukan feature engineering...")
    
    # FamilySize: jumlah anggota keluarga termasuk penumpang sendiri
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # IsAlone: apakah penumpang bepergian sendiri
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # AgeGroup: pengelompokan usia (dibuat setelah imputasi)
    # Placeholder - akan diisi setelah missing value handling
    
    # FareBin: pengelompokan harga tiket (dibuat setelah outlier handling)
    # Placeholder - akan diisi setelah outlier handling
    
    logger.info("Feature engineering selesai. Fitur baru: FamilySize, IsAlone")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values dengan strategi yang sesuai per kolom.
    
    Args:
        df: DataFrame input
        
    Returns:
        DataFrame tanpa missing values
    """
    logger.info("Menangani missing values...")
    missing_before = df.isnull().sum().sum()
    
    # Imputasi Age dengan median (robust terhadap outlier)
    if 'Age' in df.columns:
        median_age = df['Age'].median()
        df['Age'].fillna(median_age, inplace=True)
        logger.info(f"Age: diimputasi dengan median = {median_age:.1f}")
    
    # Imputasi Embarked dengan modus
    if 'Embarked' in df.columns:
        mode_embarked = df['Embarked'].mode()[0]
        df['Embarked'].fillna(mode_embarked, inplace=True)
        logger.info(f"Embarked: diimputasi dengan modus = {mode_embarked}")
    
    # Imputasi Fare dengan median (jika ada)
    if 'Fare' in df.columns and df['Fare'].isnull().sum() > 0:
        median_fare = df['Fare'].median()
        df['Fare'].fillna(median_fare, inplace=True)
        logger.info(f"Fare: diimputasi dengan median = {median_fare:.2f}")
    
    # Buat AgeGroup setelah imputasi Age
    if 'Age' in df.columns:
        df['AgeGroup'] = pd.cut(df['Age'],
                                 bins=[0, 12, 18, 35, 60, 100],
                                 labels=['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior'])
        logger.info("AgeGroup: dibuat dari kolom Age (binning)")
    
    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing_before} -> {missing_after}")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani outlier pada kolom numerik menggunakan metode IQR Capping.
    
    Args:
        df: DataFrame input
        
    Returns:
        DataFrame dengan outlier yang sudah ditangani
    """
    logger.info("Menangani outlier...")
    
    # Capping outlier pada kolom Fare
    if 'Fare' in df.columns:
        Q1 = df['Fare'].quantile(0.25)
        Q3 = df['Fare'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_count = ((df['Fare'] < lower_bound) | (df['Fare'] > upper_bound)).sum()
        df['Fare'] = df['Fare'].clip(lower=lower_bound, upper=upper_bound)
        logger.info(f"Fare: {outlier_count} outlier ditangani dengan capping [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Buat FareBin setelah outlier handling
    if 'Fare' in df.columns:
        df['FareBin'] = pd.qcut(df['Fare'], q=4,
                                  labels=['Low', 'Medium', 'High', 'Very High'],
                                  duplicates='drop')
        logger.info("FareBin: dibuat dari kolom Fare (quantile binning)")
    
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan encoding pada fitur kategorikal.
    
    Args:
        df: DataFrame input
        
    Returns:
        DataFrame dengan fitur kategorikal yang sudah di-encode
    """
    logger.info("Melakukan encoding fitur kategorikal...")
    le = LabelEncoder()
    
    # Encoding Sex
    if 'Sex' in df.columns:
        df['Sex'] = le.fit_transform(df['Sex'])
        logger.info("Sex: Label Encoded (female=0, male=1)")
    
    # Encoding Embarked
    if 'Embarked' in df.columns:
        df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))
        logger.info("Embarked: Label Encoded")
    
    # Encoding AgeGroup
    if 'AgeGroup' in df.columns:
        df['AgeGroup'] = le.fit_transform(df['AgeGroup'].astype(str))
        logger.info("AgeGroup: Label Encoded")
    
    # Encoding FareBin
    if 'FareBin' in df.columns:
        df['FareBin'] = le.fit_transform(df['FareBin'].astype(str))
        logger.info("FareBin: Label Encoded")
    
    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalisasi fitur numerik menggunakan StandardScaler.
    
    Args:
        df: DataFrame input
        
    Returns:
        DataFrame dengan fitur numerik yang sudah dinormalisasi
    """
    logger.info("Melakukan normalisasi fitur numerik...")
    scaler = StandardScaler()
    
    numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    existing_numerical = [col for col in numerical_cols if col in df.columns]
    
    df[existing_numerical] = scaler.fit_transform(df[existing_numerical])
    logger.info(f"Kolom dinormalisasi: {existing_numerical}")
    
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari dataset.
    
    Args:
        df: DataFrame input
        
    Returns:
        DataFrame tanpa duplikat
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        logger.info(f"Duplikat dihapus: {before - after} baris")
    else:
        logger.info("Tidak ada duplikat ditemukan")
    return df


def preprocess_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Pipeline lengkap preprocessing data.
    Menjalankan semua tahap preprocessing secara berurutan.
    
    Args:
        input_path: Path atau URL ke dataset mentah
        output_path: Path untuk menyimpan dataset yang sudah diproses
        
    Returns:
        DataFrame yang sudah diproses dan siap digunakan untuk training
    """
    logger.info("=" * 60)
    logger.info("MEMULAI PIPELINE PREPROCESSING")
    logger.info("=" * 60)
    
    # 1. Load data
    df = load_data(input_path)
    
    # 2. Drop kolom tidak relevan
    df = drop_irrelevant_columns(df)
    
    # 3. Feature engineering
    df = feature_engineering(df)
    
    # 4. Handle missing values
    df = handle_missing_values(df)
    
    # 5. Handle outliers
    df = handle_outliers(df)
    
    # 6. Encode categorical
    df = encode_categorical(df)
    
    # 7. Normalize features
    df = normalize_features(df)
    
    # 8. Remove duplicates
    df = remove_duplicates(df)
    
    # 9. Final validation
    logger.info("=" * 60)
    logger.info("VALIDASI AKHIR")
    logger.info(f"Shape dataset: {df.shape}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    logger.info(f"Fitur: {df.columns.tolist()}")
    
    # 10. Simpan hasil
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset preprocessing disimpan ke: {output_path}")
    logger.info("=" * 60)
    logger.info("PREPROCESSING SELESAI")
    logger.info("=" * 60)
    
    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Automate preprocessing untuk dataset Titanic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python automate_Nama-siswa.py
  python automate_Nama-siswa.py --input data/titanic_raw.csv --output data/titanic_preprocessing.csv
  python automate_Nama-siswa.py --input https://raw.githubusercontent.com/.../titanic.csv
        """
    )
    parser.add_argument(
        '--input', 
        type=str, 
        default='titanic_raw.csv',
        help='Path atau URL ke dataset mentah (default: titanic_raw.csv)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='titanic_preprocessing.csv',
        help='Path output untuk dataset yang sudah diproses (default: titanic_preprocessing.csv)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    df_result = preprocess_pipeline(
        input_path=args.input,
        output_path=args.output
    )
    print(f"\nPreprocessing berhasil! Dataset siap training tersimpan di: {args.output}")
    print(f"Shape akhir: {df_result.shape}")
