# Imported libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import os

#---------------------------------------------------

# Create output folders
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

#-------------------------------------------------------------

# Load data
file_path = "data/raw_data.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="area"
)

sns.set_theme()

#--------------------------------------------------------------------------------
# Data inspection
print(df.head())
print(df.columns.tolist())
print(df.describe())

print("\nInfo:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

#-------------------------------------------------------------

# Select variables
cols = ["cm", "area", "Feret", "Perimeter"]

#----------------------------------------------------------

# Histograms
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].hist(df["cm"], bins=10)
axes[0, 0].set_title("cm")
axes[0, 0].set_xlabel("cm")
axes[0, 0].set_ylabel("Frequency")

axes[0, 1].hist(df["area"], bins=10)
axes[0, 1].set_title("Area")
axes[0, 1].set_xlabel("Area")
axes[0, 1].set_ylabel("Frequency")

axes[1, 0].hist(df["Feret"], bins=10)
axes[1, 0].set_title("Feret")
axes[1, 0].set_xlabel("Feret Diameter")
axes[1, 0].set_ylabel("Frequency")

axes[1, 1].hist(df["Perimeter"], bins=10)
axes[1, 1].set_title("Perimeter")
axes[1, 1].set_xlabel("Perimeter")
axes[1, 1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("outputs/figures/histograms.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#---------------------------------------------------------------------------------------

# Correlation heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    df[cols].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap of Morphometric Variables")
plt.xlabel("Variables")
plt.ylabel("Variables")

plt.savefig("outputs/figures/heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#-------------------------------------------------------------

# Scatterplots between morphometric variables
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Feret",
    y="area"
)

plt.title("Feret Diameter vs Area")
plt.xlabel("Feret Diameter")
plt.ylabel("Area")

plt.savefig("outputs/figures/feret_vs_area.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Perimeter",
    y="area"
)

plt.title("Perimeter vs Area")
plt.xlabel("Perimeter")
plt.ylabel("Area")

plt.savefig("outputs/figures/perimeter_vs_area.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Feret",
    y="Perimeter"
)

plt.title("Feret Diameter vs Perimeter")
plt.xlabel("Feret Diameter")
plt.ylabel("Perimeter")

plt.savefig("outputs/figures/feret_vs_perimeter.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#-------------------------------------------------------------

# Scatterplots against age/depth proxy (cm)
fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

sns.scatterplot(
    data=df,
    x="area",
    y="cm",
    ax=axes[0]
)
axes[0].set_title("Age vs Area")
axes[0].set_xlabel("Area")
axes[0].set_ylabel("cm")

sns.scatterplot(
    data=df,
    x="Feret",
    y="cm",
    ax=axes[1]
)
axes[1].set_title("Age vs Feret")
axes[1].set_xlabel("Feret Diameter")
axes[1].set_ylabel("cm")

sns.scatterplot(
    data=df,
    x="Perimeter",
    y="cm",
    ax=axes[2]
)
axes[2].set_title("Age vs Perimeter")
axes[2].set_xlabel("Perimeter")
axes[2].set_ylabel("cm")

plt.tight_layout()
plt.savefig("outputs/figures/age_vs_morphometrics.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#-------------------------------------------------------

# Boxplot
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df[["area", "Feret", "Perimeter"]]
)

plt.title("Distribution of Morphometric Variables")
plt.xlabel("Variable")
plt.ylabel("Value")

plt.savefig("outputs/figures/boxplot.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#----------------------------------------------------------------------------

# Pairplot
pairplot = sns.pairplot(
    df[cols]
)

pairplot.fig.suptitle(
    "Pairwise Relationships Between Variables",
    y=1.02
)

pairplot.savefig(
    "outputs/figures/pairplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

#-----------------------------------------------------------

# PCA
morph_cols = ["cm", "area", "Feret", "Perimeter"]
X = df[morph_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1", "PC2"],
    index=morph_cols
)

print("\nPCA Loadings:")
print(loadings)

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2"
)

plt.title("PCA of Morphometric Variables")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.savefig("outputs/figures/pca_plot.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#-----------------------------------------------------------------------------

# Scree plot
explained = pca.explained_variance_ratio_
cum_var = np.cumsum(explained)

pcs = range(1, len(explained) + 1)

plt.figure(figsize=(8, 5))

plt.bar(
    pcs,
    explained,
    alpha=0.7,
    label="Individual variance"
)

plt.plot(
    pcs,
    cum_var,
    marker="o",
    label="Cumulative variance"
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance")
plt.title("Scree Plot")
plt.xticks(pcs)
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/figures/scree_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

#-----------------------------------------------------------------------------

# Variable summary table
variable_table = pd.DataFrame({
    "Variable": ["cm", "area", "Feret", "Perimeter"],
    "Description": [
        "Specimen age/depth proxy",
        "Shell area",
        "Maximum shell diameter",
        "Shell perimeter"
    ],
    "Type": [
        "Continuous",
        "Continuous",
        "Continuous",
        "Continuous"
    ]
})

print("\nVariable summary table:")
print(variable_table)

variable_table.to_csv(
    "outputs/tables/variable_table.csv",
    index=False
)

#--------------------------------------------------------

# Summary statistics table
summary_table = df[
    ["cm", "area", "Feret", "Perimeter"]
].describe().T

summary_table = summary_table[
    ["mean", "std", "min", "25%", "50%", "75%", "max"]
]

print("\nSummary statistics:")
print(summary_table.round(2))

print("\nSummary statistics markdown:")
print(summary_table.round(2).to_markdown())

summary_table.round(2).to_csv(
    "outputs/tables/summary_statistics.csv"
)

#---------------------------------------------------

# Correlation matrix table
corr_table = df[
    ["cm", "area", "Feret", "Perimeter"]
].corr()

print("\nCorrelation matrix:")
print(corr_table.round(3))

print("\nCorrelation matrix markdown:")
print(corr_table.round(3).to_markdown())

corr_table.round(3).to_csv(
    "outputs/tables/correlation_matrix.csv"
)

#---------------------------------------------------

# Missing values table
missing_table = pd.DataFrame({
    "Variable": df.columns,
    "Missing Values": df.isnull().sum().values
})

print("\nMissing values table:")
print(missing_table)

missing_table.to_csv(
    "outputs/tables/missing_values.csv",
    index=False
)

#-----------------------------------------------------------------

# Descriptive overview table
overview = pd.DataFrame({
    "Metric": [
        "Number of observations",
        "Number of variables",
        "Duplicate rows"
    ],
    "Value": [
        len(df),
        df.shape[1],
        df.duplicated().sum()
    ]
})

print("\nDataset overview:")
print(overview)

overview.to_csv(
    "outputs/tables/dataset_overview.csv",
    index=False
)

#--------------------------------------------------------