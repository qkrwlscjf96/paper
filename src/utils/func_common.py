from abc import ABC, abstractmethod
import pandas as pd
import os

"""파일 읽기 (CSV, Excel)"""
# 1️⃣ 추상 클래스 (공통 인터페이스)
class FileReader(ABC):
    def __init__(self, filepath: str):
        self.filepath = filepath

    @abstractmethod
    def read(self) -> pd.DataFrame:
        pass


# 2️⃣ CSV 전용 클래스
class CSVReader(FileReader):
    def read(self) -> pd.DataFrame:
        print("Reading CSV file...")
        return pd.read_csv(self.filepath).reset_index(drop=True)


# 3️⃣ Excel 전용 클래스
class ExcelReader(FileReader):
    def read(self) -> pd.DataFrame:
        print("Reading Excel file...")
        return pd.read_excel(self.filepath).reset_index(drop=True)


# 4️⃣ Factory 함수 (확장자 기반 객체 생성)
def get_reader(filepath: str) -> FileReader:
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        return CSVReader(filepath)
    elif ext in [".xlsx", ".xls"]:
        return ExcelReader(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
