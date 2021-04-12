from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def clean_decisions(data, col, percent, switch=False, chart=True):
    if chart:
        if switch:
            plot(data, col, percent, size=(2, 1))
        else:
            plot(data, col, percent)
    print('\nWhat to do? (d: drop column, s: skip, f: fill with most_frequent value')
    print('me: filled with mean value, oh: One Hot encoding, bin: Binary encoding')
    INPUT = input()
    INPUT = INPUT.replace(' ', '')
    INPUT = INPUT.split(',')
    for answer in INPUT:
        print(answer, end=': ')
        if answer == '':
            pass
        elif answer == 'd':  # Drop column
            data = data.drop(col, axis=1)
            print('Column dropped')

        elif answer == 's':  # skip and do nothing
            print('Column skipped')

        elif answer == 'f':  # fill with most_frequent value
            imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            data[col] = imp.fit_transform(pd.DataFrame(data[col]))
            print('filled with most_frequent value')

        elif answer == 'me':  # fill with mean value
            if data[col].dtypes == np.int32 or data[col].dtypes == np.float32 or data[col].dtypes == np.int64 or data[col].dtypes == np.float64:
                imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                data[col] = imp.fit_transform(pd.DataFrame(data[col]))
                print('filled with mean value')
            else:
                print('filled with mean value - FAILED!')
                print('Can\'t Impute data (non numerical values)')

        elif answer == 'oh':  # One Hot encoding
            data = data.join(pd.get_dummies(data[col], prefix=col)).drop(col, axis=1)
            print('One Hot encoding')

        elif answer == 'bin':  # binary encoding
            if data[col].nunique() == 2:
                data = data.join(pd.get_dummies(data[col], prefix=col))
                data = data.drop([f'{col}_{pd.get_dummies(data[col]).columns[0]}', col], axis=1)
                print('Binary encoding')
            elif data[col].nunique() == 1:
                print('Binary encoding FAILED! Column have 1 unique value (suggest: drop this column)')
            else:
                print('Binary encoding FAILED! To many unique values for binary encoding (suggest: OneHotEncoding)')
        else:
            print(f'{answer}: is wrong input. Break function loop.')
            break

    return data


def pre_cleaner(data):
    columns = data.columns
    for col in data.columns:
        percent = data[col].isna().sum() / data[col].shape[0]
        print('\n--------------------------------------------')
        print(f'Column: \"{col.upper()}\"', end='  ->  ')
        print('No. unique values:', data[col].nunique(), '; Missing data:', percent, '% ; dtype:', data[col].dtypes, '\n')
        # print(data[col].describe())
        data_type = (data[col].dtypes == np.int32 or
                     data[col].dtypes == np.float32 or
                     data[col].dtypes == np.int64 or
                     data[col].dtypes == np.float64)

        if data[col].isna().sum() == 0 and data_type:
            print('Data are correct in this column.')
        elif data[col].isna().sum() != 0 and data_type:
            print('Data are correct format but have empty spots.')
            data = clean_decisions(data, col, percent, chart=False)
        else:
            if data[col].nunique() > 10:
                while True:
                    ans = input(f'Many unique values! ({data[col].nunique()}) Do you want to print data on the graph?(y/n)')
                    if ans == ('y' or 'Y'):
                        data = clean_decisions(data, col, percent)
                        break
                    elif ans == ('n' or 'N'):
                        data = clean_decisions(data, col, percent, chart=False)
                        # print('n/N: Column skipped')
                        break
            else:
                data = clean_decisions(data, col, percent)

    data.to_csv('pre_cleaned_data.csv')
    print('Data exported to file. (pre_cleaned_data.csv)')


def plot(data, col, percent, size=(1, 2)):
    label = data[col].value_counts().index
    fig, (ax1, ax2) = plt.subplots(size[0], size[1])
    ax1.bar(label, data[col].value_counts().values)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
    ax2.pie(data[col].value_counts().values, labels=label, autopct='%1.1f%%')
    fig.suptitle(f'Column: \'{col}\'   -   Missing data: {percent:.2f}% ({data[col].isna().sum()})', fontsize=15)
    fig.tight_layout()
    plt.show()


def to_clean(data):
    columns = data.columns
    for col in data.columns:
        data_type = (data[col].dtypes == np.int32 or data[col].dtypes == np.float32 or
                     data[col].dtypes == np.int64 or data[col].dtypes == np.float64)

        if data_type == False or data[col].isna().sum() != 0:
            print(f'\"{col}\"', end=', ')


if __name__ == '__main__':
    # load example data:  https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists

    FILE_NAME = 'aug_train.csv'     # Name of file to clean

    df = pd.read_csv(FILE_NAME)
    # df = df.iloc[:1000, :]
    print(f'Data loaded from file: \"{FILE_NAME}\"\n')
    for x in df.columns:
        print(x, end=', ')
    print('')
    print(f'Number of columns: {df.shape[1]}')
    input('\nWait for key to start')

    pre_cleaner(df)  # it export cleaned data automatically

