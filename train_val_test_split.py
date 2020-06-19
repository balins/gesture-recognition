import os
import shutil
import numpy as np

train = "data/train"
val = "data/val"
test = "data/test"

count_total, count_train, count_val, count_test = 0, 0, 0, 0

for directory in 'NONE OKAY OPEN_HAND PEACE POINTING SHAKA THUMBS_UP'.split():
    os.system('mkdir -p {0}/{1}'.format(val, directory))
    os.system('mkdir -p {0}/{1}'.format(test, directory))

    files = os.listdir('{0}/{1}'.format(train, directory))
    count_total += len(files)
    for file in files:
        if np.random.rand(1) < 0.3:
            shutil.move('{0}/{1}/{2}'.format(train, directory, file),
                        '{0}/{1}/{2}'.format(val, directory, file))
            count_val += 1
        elif np.random.rand(1) < 0.02:
            shutil.move('{0}/{1}/{2}'.format(train, directory, file),
                        '{0}/{1}/{2}'.format(test, directory, file))
            count_test += 1
        else:
            count_train += 1

print("""
    TOTAL SIZE: {0} SAMPLES ({1}% OF DATASET),
    TRAIN DATA: {2} SAMPLES ({3}% OF DATASET),
    VALIDATION DATA: {4} SAMPLES ({5}% OF DATASET),
    TEST DATA: {6} SAMPLES ({7}% OF DATASET)
""".format(count_total, 100,
           count_train, (count_train * 100.0 / count_total).__round__(2),
           count_val, (count_val * 100.0 / count_total).__round__(2),
           count_test, (count_test * 100.0 / count_total).__round__(2))
      )
