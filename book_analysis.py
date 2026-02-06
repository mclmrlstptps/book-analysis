import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset and Skip bad lines
df = pd.read_csv("books.csv", on_bad_lines="skip")

# Display basic dataset info
print("Initial Dataset Info:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())

# Clean data and convert average rating and number of pages to a float
df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
df['  num_pages'] = pd.to_numeric(df['  num_pages'], errors='coerce')

# Remove books with 0 pages
df = df[(df['  num_pages'] > 0) & (df['  num_pages'] <= 2000)]

# Remove rows missing critical values
df = df.dropna(subset=['average_rating', '  num_pages', 'authors'])

# Strip Whitespace from author names
df['authors'] = df['authors'].str.strip()

# Verify that data was cleaned successfully
print("\nCleaned Dataset Info:")
print(df.info())


# Summary of Statistics 
print("\nSummary Statistics for Ratings:")

# .describe() calculates
# count -> number of valid ratings
# mean -> average rating
# std -> standard deviation (how spread out ratings are)
# min -> lowest rating
# 25%, 50%, 75% -> quartiles (data distribution)
# max -> highest rating
print(df['average_rating'].describe())

# .describe for page count
print("\nSummary Statistics for Page Counts:")
print(df['  num_pages'].describe())

# Question 1 
# What author has the highest average rating? 

# Group by Author
author_stats = df.groupby('authors').agg(
    avg_rating=('average_rating', 'mean'),
    book_count=('average_rating', 'count')
)

# Filter Authors with more than 5 books
author_stats = author_stats[author_stats['book_count'] >= 5]

# Sort by average rating
author_stats = author_stats.sort_values(by='avg_rating', ascending=False)

# Display top authors
print("\nTop 10 Authors by Average Rating:")
print(author_stats.head(10))

# Graph Question 1 - layout display
top_authors = author_stats.head(10)

plt.figure()
top_authors['avg_rating'].plot(kind='bar')
plt.title("Top Authors by Average Rating (Min 2 Books)")
plt.xlabel("Author")
plt.ylabel("Average Rating")
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()

# Question 2
# Define page length bins
bins = [0, 200, 400, 600, 1000]
labels = ['<200', '200–400', '400–600', '600+']

# Categorize books by length
df['page_group'] = pd.cut(df['  num_pages'], bins=bins, labels=labels)

# Calculate average rating per page group
page_rating = df.groupby('page_group')['average_rating'].mean()

# Display results
print("\nAverage Rating by Page Length:")
print(page_rating)

# Question 2 Graph - layout display
plt.figure()
page_rating.plot(kind='bar')
plt.title("Average Rating by Book Length")
plt.xlabel("Page Length Group")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.show()

# Final interpretation output
print("\nAnalysis Complete.")
print("\nFinal Summary")

# Question 2 Summary
print(
    "Author Analysis:\n"
    "When books grouped by author and while calculating the average rating,"
    "the results show that authors with multiple books tend to have more"
    "Stable and higher average ratings."
)

# Question 2 Summary
print(
    "\nBook length analysis:\n"
    "When books were grouped by length, the average ratings showed "
    "that medium-length books recieved slightly higher ratings"
    "than short or long books. Suggesting that readers may prefer" 
    "books that are long enough for a good story, without being too lengthy"
)
