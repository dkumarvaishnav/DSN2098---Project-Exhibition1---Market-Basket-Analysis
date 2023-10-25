import pandas as pd

# Dictionary with movie titles and genres
movie_genres = {
    "The Irishman": "Crime, Drama, Biography",
    "Dangal": "Biography, Drama, Sport",
    "David Attenborough: A Life on Our Planet": "Documentary",
    "Lagaan: Once Upon a Time in India": "Drama, Sport, Musical",
    "Roma": "Drama",
    "To All the Boys I've Loved Before": "Romance, Comedy, Teen",
    "The Social Dilemma": "Documentary, Drama",
    "Okja": "Adventure, Drama, Sci-Fi",
    "The Ballad of Buster Scruggs": "Western, Anthology, Comedy",
    "The Trial of the Chicago 7": "Drama, Historical",
    "Article 15": "Crime, Drama, Thriller",
    "Jim & Andy: The Great Beyond- Featuring a Very Special, Contractually Obligated Mention of Tony Clifton": "Documentary, Comedy, Biography",
    "Dolemite Is My Name": "Biography, Comedy, Drama",
    "Mudbound": "Drama, Historical",
    "Swades": "Drama",
    "Fyre": "Documentary",
    "Miss Americana": "Documentary, Music",
    "Virunga": "Documentary",
    "Black Friday": "Crime, Drama, Thriller",
    "Talvar": "Crime, Drama, Mystery",
}

# Load the existing CSV file
existing_file = 'SelectedMovies.csv'
df = pd.read_csv(existing_file)

# Add a new 'Genre' column with empty values to the DataFrame
df['Genre'] = ''

# Fetch genre information for each movie title from the dictionary and update the 'Genre' column
for index, row in df.iterrows():
    title = row['Title']
    if title in movie_genres:
        genre_string = movie_genres[title]
        df.at[index, 'Genre'] = genre_string

# Save the updated DataFrame to a new CSV file
output_file = 'final.csv'
df.to_csv(output_file, index=False)

print(f"Genres have been added and saved to '{output_file}'.")
