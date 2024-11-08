import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import sys

ST_COL = 'st'

def _load_qualtrics_form(form_csv_file, slots=6):
    """Which professors do you would like to meet during speed advising? Please pick at least 10 faculty from the list. - Adam Bates (UIUC) -- Intrusion Detection, System Security, Usable Security"""
    global ST_COL
    IGNORE_FIRST_N_COLUMNS = 3
    d = pd.read_csv(form_csv_file)
    qcols = d.columns[d.columns.str.startswith("Q")]
    prof_cols = d.columns[d.columns.str.startswith("Q3")]
    profs = d[prof_cols].iloc[0].str.split('-').apply(lambda x: x[1].split('-')[0].strip())

    d = d[qcols].iloc[IGNORE_FIRST_N_COLUMNS:].fillna('No Selection')
    label_map = {
        'Extremely interested in meeting': 1,
        'Very interested in meeting': 2,
        'Interested in meeting': 3,
        'Somewhat interested in meeting': 4,
        'No Selection': 2
    }
    d[qcols] = d[qcols].applymap(lambda x: label_map.get(x, x))
    qcols = list(qcols[:1]) + list(qcols[2:])
    a = d[qcols].reset_index(drop=True)
    a.rename(columns={'Q1': ST_COL}, inplace=True)
    a.drop('Q6', axis=1, inplace=True)
    print(a)

    ## Chunk 1
    _nstudents_by2 = len(a)//2
    achunk1 = a.iloc[:_nstudents_by2].reset_index(drop=True).copy()
    achunk2 = a.iloc[_nstudents_by2:].reset_index(drop=True).copy()
    od = pd.DataFrame({'prof': profs})
    prof_sch = np.concatenate(
        [get_schedule(achunk1, slots=slots//2), get_schedule(achunk2, slots=slots//2)+_nstudents_by2],
        axis=1
    )
    students = a[ST_COL]
    students.loc[len(students)] = 'Not-Assigned'
    for s in range(slots):
        st_indices = prof_sch[:, s]
        st_indices[st_indices<0] = len(students)-1
        print(st_indices, students)
        od[f"slot{s}"] = students[st_indices].to_numpy()
        print(od)

    od.to_csv(f"schedule_{form_csv_file}", index=False)
    return a

## d is a pandas dataframe rows are students and columns are professors
def get_schedule(d, slots=4):
    all_profs = d.columns[1:]
    nstudent, nprof = d.shape[0], d.shape[1] - 1
    prof_sch = np.zeros((nprof, slots), dtype=int) - 999
    st_meet_cnt = np.zeros(nstudent, dtype=int)
    
    approx_match_per_student = min(int(nprof * slots / nstudent + 0.5), slots)
    print(f"Each student should meet around: {approx_match_per_student} profs")
    # Preference matrix
    a = d.to_numpy()[:, 1:].astype(int)
    for i in range(slots):
        row_inds, col_inds = linear_sum_assignment(a)
        a[row_inds, col_inds] = 999
        print(d[ST_COL], row_inds)
        print(f"Match-{i+1}: {list(zip(row_inds, d[ST_COL][row_inds], d.columns[col_inds+1]))}")
        st_meet_cnt[row_inds] += 1
        prof_sch[col_inds, i] = row_inds
        _ismeeting_toomany, = np.where(st_meet_cnt>approx_match_per_student)
        if len(_ismeeting_toomany)>0:
            a[_ismeeting_toomany, :] = 999
            print(f"These students are meeting more than {approx_match_per_student} profs:")
            print(f"{d[ST_COL][_ismeeting_toomany]}")
            # print(a)
    print(f"Meet counts: {st_meet_cnt}")
    assert (st_meet_cnt>0).all(), "Some students are not meeting any professors"
    assert (st_meet_cnt<=(approx_match_per_student+1)).all(), "Some students are meeting too many professors"
    return prof_sch

def verify_and_export(d, prof_sch, outf_name="schedule.csv"):
    """
    matches is a dictionary from a student to a list of prof
    """
    nprof, slots = prof_sch.shape
    print("=="*30)
    print(f"Schedule for {d.shape[0]} students among {d.shape[1]-1} professors in {slots} slots .")
    print("=="*30)
    for i in range(nprof):
        p = d.columns[i+1]
        s = prof_sch[i][prof_sch[i]>-1]
        sts = '\t'.join(d[ST_COL][s])
        print(f"{p} => \t {sts}")
        np.testing.assert_equal(len(np.unique(s)), len(s), f"Prof {p} meeting same student twice {sts}.")
    for s in range(slots):
        st_involved = prof_sch[:, s][prof_sch[:, s] > 0]
        np.testing.assert_equal(len(np.unique(st_involved)), len(st_involved), f"Some students are meeting multuiple faculty in the same slot {s}: {st_involved}.")

def test_scheduling():
    d = pd.read_csv('sp.csv')
    # print(d)
    x = list(np.arange(d.shape[1]-1)+1)
    # np.random.shuffle(x)
    d = d.iloc[:, [0] + x]
    print(d)
    slots = 4
    matches = get_schedule(d, slots)
    verify(d, matches)
    
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Need to provide the qualtrics export file name to load the data from")
        exit(1)
    d = _load_qualtrics_form(sys.argv[1])
