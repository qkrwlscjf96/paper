import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_boxplots_by_date(df:pd.DataFrame, check_cols:list)->None:
    """용해탱크 데이터에 대한 날짜별 박스플롯 시각화 함수"""
    
    for col in check_cols:
        plt.figure(figsize=(14, 6))

        sns.scatterplot(
            data=df,
            x="DATE",
            y=col,
            hue="TAG",
            alpha=0.7
        )

        plt.title(f"{col} Scatter Plot by Date (TAG)")
        plt.xlabel("Date")
        plt.ylabel(col)

        # x축 90도 회전
        plt.xticks(rotation=90)

        # legend 오른쪽 위 고정
        plt.legend(title="TAG", loc="upper right")

        plt.tight_layout()
        plt.show()
