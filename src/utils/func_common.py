from abc import ABC, abstractmethod
import pandas as pd
import os
from pathlib import Path

"""파일 읽기 (CSV, Excel)"""
# 1추상 클래스 (공통 인터페이스)
class FileReader(ABC):
    def __init__(self, filepath: str):
        self.filepath = filepath

    @abstractmethod
    def read(self) -> pd.DataFrame:
        pass


# CSV 전용 클래스
class CSVReader(FileReader):
    def read(self) -> pd.DataFrame:
        print("Reading CSV file 완료")
        return pd.read_csv(self.filepath).reset_index(drop=True)


# Excel 전용 클래스
class ExcelReader(FileReader):
    def read(self) -> pd.DataFrame:
        print("Reading Excel file 완료")
        return pd.read_excel(self.filepath).reset_index(drop=True)


# Factory 함수 (확장자 기반 객체 생성)
def get_reader(filepath: str) -> FileReader:
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        return CSVReader(filepath)
    elif ext in [".xlsx", ".xls"]:
        return ExcelReader(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

"""실제 파일 읽기"""

def load_data_df(data_name: str, base_path: Path):
    """
    data_name : '용해탱크', '사출성형기', '살균기', '소성가공'
    return : df, ng_df, target_col, check_cols
    """

    data_path = base_path / "data"
    file_paths = os.listdir(data_path)

    # 파일 찾기
    pick_file = [f for f in file_paths if data_name in f][0]
    file_path = data_path / pick_file

    df = get_reader(file_path).read()

    # 날짜 컬럼 처리 (STD_DT 예외 처리)
    if data_name != "소성가공":
        df["DATE"] = pd.to_datetime(df["DATE"]).dt.normalize() # yyyy-mm-dd 00:00:00 형태로 변환
    else:
        df["DATE"] = pd.to_datetime(df["DATE"]).dt.floor("min") # yyyy-mm-dd HH:MM:00 형태로 변환

    # 기본 설정
    target_col = ["TAG"]
    date_col = ["DATE"]
    check_cols = [x for x in df.columns if x not in target_col + date_col]

    # TAG → 1/0 변환
    df[target_col[0]] = (df[target_col[0]] == "NG").astype(int)

    # 결측치 처리 (수치형만 평균 대체)
    df = df.fillna(df.mean(numeric_only=True))

    # NG 데이터
    ng_df = df[df[target_col[0]] == 1]

    return df, ng_df, target_col, check_cols, date_col