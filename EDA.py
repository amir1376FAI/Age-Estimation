import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


import cv2 as cv
import PIL
from PIL import Image

import seaborn as sns

#  *********************************** Download & Show Images  ***********************************
# Replace with the actual path to your UTK dataset images folder
dataset_folder = '/content/UTKFace/'

def show_random_samples(num_samples=9):
    image_files = os.listdir(dataset_folder)
    selected_image_files = random.sample(image_files, num_samples)

    plt.figure(figsize=(10, 10))
    for idx, image_file in enumerate(selected_image_files, 1):
        image_path = os.path.join(dataset_folder, image_file)
        age, gender, ethnicity = image_file.split('_')[:3]

        image = Image.open(image_path)

        gender = 'Male' if int(gender) == 0 else 'Female'
        ethnicity = ['White', 'Black', 'Asian', 'Indian', 'Others'][int(ethnicity)]

        plt.subplot(3, 3, idx)
        plt.imshow(image)
        plt.title(f"Age: {age}\nGender: {gender}\nEthnicity: {ethnicity}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Call the function to display random samples
show_random_samples()

#  *********************************** Create a csv file which contains labels  ***********************************
dataset_folder = '/content/UTKFace/'
image_files = os.listdir(dataset_folder)
info = []
test = []
index_to_remove = []

for idx, image_file in enumerate(image_files):
    test.append(image_file.split('_'))
for idx,t in enumerate(test):
    if len(t) != 4:
        index_to_remove.append(idx)


for idx, image_file in enumerate(image_files):
    if idx in index_to_remove:
        continue

    tmp = dict()
    age, gender, ethnicity = image_file.split('_')[:3]
    gender = 'Male' if int(gender) == 0 else 'Female'
    ethnicity = ['White', 'Black', 'Asian', 'Indian', 'Others'][int(ethnicity)]

    if  int(age) > 80:
        continue

    tmp['image_name'] = image_file
    tmp['age'] = int(age)
    tmp['ethnicity'] = ethnicity
    tmp['gender'] = gender
    info.append(tmp)



df = pd.DataFrame(info)
df.to_csv('/content/utkface_dataset.csv')
df

# *********************************** Plot histogram for age  ***********************************
plt.figure(figsize=(16, 8))
ax = sns.histplot(data=df, x='age', bins=30, kde=True,
                 edgecolor='white', linewidth=0.5,
                 color='#4C72B0', alpha=0.8)

# Customization
plt.title('Age Distribution\n', fontsize=16, fontweight='bold')
plt.xlabel('Age (years)', fontsize=12, labelpad=15)
plt.ylabel('Count', fontsize=12, labelpad=15)

# Format axes
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add annotations
mean_age = df['age'].mean()
median_age = df['age'].median()
plt.axvline(mean_age, color='red', linestyle='--', linewidth=1)
plt.axvline(median_age, color='green', linestyle='--', linewidth=1)

plt.text(0.95, 0.95,
         f"Total samples: {len(df):,}\nMean age: {mean_age:.1f}\nMedian age: {median_age:.1f}",
         transform=ax.transAxes, ha='right', va='top',
         bbox=dict(facecolor='white', alpha=0.8))

# Final polish
sns.despine()
plt.tight_layout()
plt.show()

# *********************************** Plot histogram for gender ***********************************


# Set style and context first (affects all following plots)
sns.set_theme(style="whitegrid", context="notebook")

# Create figure and axes objects explicitly
plt.figure(figsize=(12, 7))

# Order categories by frequency (descending)
gender_order = df['gender'].value_counts().index

# Create count plot with ordered categories
ax = sns.countplot(
    x='gender',
    data=df,
    order=gender_order,
    edgecolor='black',
    linewidth=1,
    palette='Set2',
    hue='gender'
)

# Customization
plt.title('Distribution of Gender\n', fontsize=16, fontweight='bold')
plt.xlabel('Gender', fontsize=12, labelpad=15)
plt.ylabel('Count', fontsize=12, labelpad=15)

# Format y-axis with thousands separators
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

# Add data labels
for p in ax.patches:
    ax.annotate(
        f'{p.get_height():,.0f}',
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center',
        va='center',
        xytext=(0, 5),
        textcoords='offset points',
        fontsize=10
    )

# Rotate x-axis labels
plt.xticks(
    rotation=45,
    ha='right',
    fontsize=11,
    rotation_mode='anchor'
)

# Adjust grid and borders
ax.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine(left=True, bottom=True)

# Add descriptive text if needed
plt.text(
    x=0.5,
    y=-0.25,
    s=f'Total samples: {len(df):,}',
    transform=ax.transAxes,
    ha='center',
    fontsize=10,
    color='grey'
)

# Tight layout and show
plt.tight_layout()
plt.show()

# *********************************** Plot histogram for ethnicity ***********************************


# Set style and context first (affects all following plots)
sns.set_theme(style="whitegrid", context="notebook")

# Create figure and axes objects explicitly
plt.figure(figsize=(12, 7))

# Order categories by frequency (descending)
ethnicity_order = df['ethnicity'].value_counts().index

# Create count plot with ordered categories
ax = sns.countplot(
    x='ethnicity',
    data=df,
    order=ethnicity_order,
    edgecolor='black',
    linewidth=1,
    palette='Set2',
    hue='ethnicity'
)

# Customization
plt.title('Distribution of Ethnicities\n', fontsize=16, fontweight='bold')
plt.xlabel('Ethnicity', fontsize=12, labelpad=15)
plt.ylabel('Count', fontsize=12, labelpad=15)

# Format y-axis with thousands separators
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

# Add data labels
for p in ax.patches:
    ax.annotate(
        f'{p.get_height():,.0f}',
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center',
        va='center',
        xytext=(0, 5),
        textcoords='offset points',
        fontsize=10
    )

# Rotate x-axis labels
plt.xticks(
    rotation=45,
    ha='right',
    fontsize=11,
    rotation_mode='anchor'
)

# Adjust grid and borders
ax.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine(left=True, bottom=True)

# Add descriptive text if needed
plt.text(
    x=0.5,
    y=-0.25,
    s=f'Total samples: {len(df):,}',
    transform=ax.transAxes,
    ha='center',
    fontsize=10,
    color='grey'
)

# Tight layout and show
plt.tight_layout()
plt.show()

# *********************************** Create violin plots and box plots for age, separately for each ethnicity ***********************************


# Set professional style
sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(12, 8))

# Create violin plot with corrected hue/palette usage
ax = sns.violinplot(
    x="age",
    y="ethnicity",
    hue="ethnicity",  # <- Critical fix here
    data=df,
    inner=None,
    palette="pastel",
    saturation=0.7,
    linewidth=1.5,
    width=0.8,
    legend=False  # <- Disable redundant legend
)

# Overlay boxplot with contrasting style
sns.boxplot(
    x="age",
    y="ethnicity",
    data=df,
    width=0.15,
    ax=ax,
    color="black",
    linewidth=2,
    flierprops={
        'marker': 'D',
        'markerfacecolor': 'red',
        'markersize': 8,
        'markeredgecolor': 'none'
    }
)

# (Rest of the styling code remains the same)
plt.title("Age Distribution by Ethnicity", fontsize=20, pad=20, weight='bold')
plt.xlabel("Age", fontsize=16, labelpad=15)
plt.ylabel("Ethnicity", fontsize=16, labelpad=15)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(True, axis='x', linestyle='--', alpha=0.7)

for i, ethnicity in enumerate(df['ethnicity'].unique()):
    count = df[df['ethnicity'] == ethnicity]['age'].count()
    ax.text(x=0.95, y=i, s=f'n = {count}',
            ha='right', va='center',
            fontsize=14, color='darkblue',
            transform=ax.get_yaxis_transform())

sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()

               
