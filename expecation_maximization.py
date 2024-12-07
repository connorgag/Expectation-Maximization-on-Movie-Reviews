import pandas as pd
import math

def process_movies():
    with open('movies-1.txt', 'r') as file:
        movies = []
        for line in file:
            movies.append(line.replace('\n', ''))
    return movies

def process_ids():
    with open('ids-2.txt', 'r') as file:
        ids = []
        for line in file:
            ids.append(line.replace('\n', ''))
    return ids


def process_ratings():
    with open('ratings-1.txt', 'r') as file:
        ratings = []
        for line in file:
            ratings.append([i.strip() for i in line.split()])
    return ratings

def process_prob_z():
    initial_z_values = []
    with open('probZ_init.txt', 'r') as file:
        for line in file:
            initial_z_values.append(float(line.replace('\n', '').replace(' ', '')))
    return initial_z_values


def process_prob_r():
    initial_r_values = []
    with open('probR_init.txt', 'r') as file:
        for line in file:
            initial_r_values.append([float(i) for i in (line.replace('\n', '').split())])
    return initial_r_values


def create_ratings_df():
    df = pd.DataFrame(process_ratings(), columns=process_movies())

    likes_per_column = df.apply(lambda col: (col == '1').sum())
    watches_per_column = df.apply(lambda col: (col != '?').sum())
    likes_to_watches_ratio = (likes_per_column / watches_per_column)

    print(likes_to_watches_ratio.sort_values())

    return df


# Given a student's ratings, compute the posterior probability
def compute_posterior_prob(given_ratings, all_students_ratings, z_values, r_values):
    # For each possible value of Z, find the product in the num and den
    product_for_each_i = [1] * len(z_values)
    for i in range(len(product_for_each_i)):
        # Go through all movies for this student
        for j in range(len(given_ratings)):
            # Don't consider case ? because it's unknown
            if (given_ratings[j] == '1'):
                product_for_each_i[i] = product_for_each_i[i] * r_values[j][i]
            elif (given_ratings[j] == '0'):
                product_for_each_i[i] = product_for_each_i[i] * (1 - r_values[j][i])
    # print(product_for_each_i)          
    # print(z_values)

    numerator = [z_values[i] * product_for_each_i[i] for i in range(len(z_values))]

    denominator = 0
    for k in range(len(z_values)):
        denominator += (z_values[k] * product_for_each_i[k])
    
    # print(denominator)
    result = [numerator[i] / denominator for i in range(len(z_values))]
            
    return result


# Compute the log likelihood P(R1, R2, ..., Rj) = P(R1)P(R2)...P(Rj) by naive bayes assumption
def compute_log_likelihood(z_values, r_values, this_student_ratings):
    z_sum = 0
    for i in range(len(z_values)):
        product = 1
        for j in range(len(r_values)):
            if (this_student_ratings[j] == '1'):
                product *= r_values[j][i]
            elif (this_student_ratings[j] == '0'):
                product *= (1.0 - r_values[j][i])
        z_sum += product * z_values[i]
    return math.log(z_sum)      


def em(ratings_df, r_values, z_values, iterations):
    # P(Z=i) = z_values[i] (0 indexed)
    # P(Rj=1 | Z=i) = r_values[j][i] (0 indexed)

    # For each iteration
    for i in range(iterations):
        # Compute log likelihood
        log_sum = 0
        for index, row in ratings_df.iterrows():
            this_student_ratings = list(row)
            # Sum up the log likelihood of this student's ratings
            log_sum += compute_log_likelihood(z_values, r_values, this_student_ratings)
        # Then divide by T to get the average
        
        log_likelihood = log_sum / len(ratings_df)
        if (i in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)):
            print("Iteration " + str(i) + " log likelihood " + str(log_likelihood))


        # print("E Step")
        pti = []
        # For each student
        for index, row in ratings_df.iterrows():
            this_student_ratings = list(row)
            # Compute E-Step posterior probability Pit (store in matrix)
            # P(Z=i | student t's ratings)
            pti.append(compute_posterior_prob(this_student_ratings, ratings_df, z_values, r_values))
            
        
        # Compute M-Step updates
        # Update CPT for P(Z=i)
        new_z_values = [None for _ in range(len(z_values))]
        # print("M Step: P(Z=i)")
        for i in range(len(z_values)):
            z_sum = 0
            for student_index in range(len(ratings_df)):
                z_sum += pti[student_index][i]
            new_z_values[i] = z_sum/len(ratings_df)


        # Update CPT for P(Rj=1 | Z=i)
        # print("M Step: P(Rj=1 | Z=i)")
        # Create a new r_vaues (can't overwrite in place)
        new_r_values = [[None for i in  range(len(z_values))] for j in range(len(r_values))]
        for i in range(len(z_values)):
            # Compute denominator
            pti_sum = 0
            # For each student
            # print("student loop")
            for student_index in range(len(ratings_df)):
                pti_sum += pti[student_index][i]

            # For each movie Rj
            for j in range(len(r_values)):
                num_first_sum = 0
                num_second_sum = 0
                # For each student
                for student_index, row in ratings_df.iterrows():
                    this_student_ratings = list(row)
                    # If the student has seen move j
                    if (this_student_ratings[j] != '?'):
                        num_first_sum += (pti[student_index][i] * (this_student_ratings[j] == '1'))
                    # If the student has not seen move j
                    else:
                        num_second_sum += (pti[student_index][i] * r_values[j][i])

                new_r_values[j][i] = (num_first_sum + num_second_sum) / pti_sum

        z_values = new_z_values
        r_values = new_r_values

    return  r_values, z_values


def compute_personal_recommendation(student_id, ratings_df, r_values, z_values):
    student_movies = ratings_df.loc[process_ids().index(student_id)]
    movie_list = list(student_movies.index)
    student_movies = list(student_movies)

    posterior_prob = compute_posterior_prob(student_movies, ratings_df, z_values, r_values)

    unseen_movies_and_ratings = []
    for j in range(len(student_movies)):
        z_sum = 0
        if (student_movies[j] == '?'):
            for i in range(len(z_values)):
                z_sum += posterior_prob[i] * r_values[j][i]
            unseen_movies_and_ratings.append([movie_list[j], z_sum])

    df = pd.DataFrame(unseen_movies_and_ratings, columns=['Unseen Movie', 'Expected Rating'])
    print(df.sort_values(by=['Expected Rating']).reset_index().drop('index', axis=1))


def main():
    ratings_df = create_ratings_df()
    r_values, z_values = em(create_ratings_df(), process_prob_r(), process_prob_z(), 257)

    compute_personal_recommendation('ID_221', ratings_df, r_values, z_values)


main()