# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 23:40:48 2022

@author: Adela Hlobilova, adela.hlobilova@gmail.com
"""

def round_scores(student_scores):
    """"Round all provided student scores.

    :param student_scores: list of student exam scores as float or int.
    :return: list of student scores *rounded* to nearest integer value.
    """

    ## classical for loop
    # rounded_scores = []
    # for score in student_scores:
    #     rounded_scores.append(round(score))
    # return rounded_scores
    
    ## this method needs definition of rounded_scores since append returns None
    # rounded_scores = []
    # [rounded_scores.append(round(score)) for score in student_scores]
    # return rounded_scores

    ## list comperhensions
    return [round(score) for score in student_scores]

def count_failed_students(student_scores):
    """Count the number of failing students out of the group provided.

    :param student_scores: list of integer student scores.
    :return: integer count of student scores at or below 40.
    """

    ## classical for loop
    # failed = 0
    # for score in student_scores:
    #     if score <= 40:
    #         failed += 1
            
    # return failed
    
    ## lenght of failed_students array via list comperhensions
    return len([score for score in student_scores if score <= 40])


def above_threshold(student_scores, threshold):
    """Determine how many of the provided student scores were 'the best' based on the provided threshold.

    :param student_scores: list of integer scores
    :param threshold :  integer
    :return: list of integer scores that are at or above the "best" threshold.
    """

    ## classical for loop
    # students_above_threshold = []
    # for score in student_scores:
    #     if score >= threshold:
    #         students_above_threshold.append(score)
            
    # return students_above_threshold

    return [score for score in student_scores if score >= threshold]

def letter_grades(highest):
    """Create a list of grade thresholds based on the provided highest grade.

    :param highest: integer of highest exam score.
    :return: list of integer lower threshold scores for each D-A letter grade interval.
             For example, where the highest score is 100, and failing is <= 40,
             The result would be [41, 56, 71, 86]:

             41 <= "D" <= 55
             56 <= "C" <= 70
             71 <= "B" <= 85
             86 <= "A" <= 100
    """

    # increment = (highest - 40 - 4) / 4
    # lower_threshold_scores = [41,]
    # for _ in range(3):
    #     lower_threshold_scores.append(int(lower_threshold_scores[-1]+increment+1))
    # return lower_threshold_scores

    # return [41 + (int((highest - 40) / 4))*i for i in range(0,4)]
    
    return list(range(41, highest, int((highest - 40) / 4)))


def student_ranking(student_scores, student_names):
    """Organize the student's rank, name, and grade information in ascending order.

     :param student_scores: list of scores in descending order.
     :param student_names: list of names in descending order by exam score.
     :return: list of strings in format ["<rank>. <student name>: <score>"].
     """

    ## classical for loop
    # out = []
    
    # for i, score in enumerate(student_scores):
    #     out.append(f"{i+1}. {student_names[i]}: {score}")

    # return out
    
    ## list comperhension
    return [f"{i+1}. {student_names[i]}: {score}" for i, score in enumerate(student_scores)]

def perfect_score(student_info):
    """Create a list that contains the name and grade of the first student to make a perfect score on the exam.

    :param student_info: list of [<student name>, <score>] lists
    :return: first `[<student name>, 100]` or `[]` if no student score of 100 is found.
    """

    ## classical for loop
    # for info in student_info:
    #     if info[1] == 100:
    #         return info
    # return []
    
    # thing_index = thing_list.index(elem) if elem in thing_list else -1

    ## using list comperhension for filtering scores and ternary operator
    # scores = [info[1] for info in student_info]
    # return student_info[scores.index(100)] if 100 in scores else []

    return student_info[scores.index(100)] if 100 in (scores:=[info[1] for info in student_info]) else []

student_scores = [90.33, 40.5, 55.44, 70.05, 30.55, 25.45, 80.45, 95.3, 38.7, 40.3]
print(round_scores(student_scores))

print(count_failed_students(student_scores=[90,40,55,70,30,25,80,95,38,40]))
print(above_threshold(student_scores=[90,40,55,70,30,68,70,75,83,96], threshold=75))
print(letter_grades(highest=100))
print(letter_grades(highest=88))
student_scores = [100, 99, 90, 84, 66, 53, 47]
student_names =  ['Joci', 'Sara','Kora','Jan','John','Bern', 'Fred']
print(student_ranking(student_scores, student_names))
print(perfect_score(student_info=[["Charles", 90], ["Tony", 80], ["Alex", 100]]))
print(perfect_score(student_info=[["Charles", 90], ["Tony", 80]]))
print(perfect_score([['Yoshi', 52], ['Jan', 86], ['Raiana', 100], ['Betty', 60], ['Joci', 100], ['Kora', 81], ['Bern', 41], ['Rose', 94]]))