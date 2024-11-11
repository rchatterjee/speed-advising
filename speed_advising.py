import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import sys
import random

ST_COL = 'st'

def _load_qualtrics_form(form_csv_file, slots=6):
    """Which professors do you would like to meet during speed advising? Please pick at least 10 faculty from the list. - Adam Bates (UIUC) -- Intrusion Detection, System Security, Usable Security"""
    global ST_COL
    IGNORE_FIRST_N_ROWS = 2
    d = pd.read_csv(form_csv_file)
    qcols = d.columns[d.columns.str.startswith("Q")]
    prof_cols = d.columns[d.columns.str.startswith("Q3")]
    RND_IDX = np.random.permutation(np.arange(len(prof_cols)))
    prof_cols = prof_cols[RND_IDX]
    profs = d[prof_cols].iloc[0].str.split('-', n=1).apply(lambda x: x[1].split('--', 1)[0].strip())
    d = d[qcols].iloc[IGNORE_FIRST_N_ROWS:].fillna('No Selection')
    label_map = {
        'Extremely interested in meeting': 1,
        'Very interested in meeting': 2,
        'Interested in meeting': 3,
        'Somewhat interested in meeting': 4,
        'No Selection': 5
    }
    d[prof_cols] = d[prof_cols].map(lambda x: label_map.get(x, x))
    res = pd.DataFrame({
        'prof': profs,
        'pref': (d[prof_cols]<=2).sum(axis=0)
    })
    qcols = np.concatenate(([qcols[0]], prof_cols))
    a = d[qcols].reset_index(drop=True)
    a.rename(columns={'Q1': ST_COL}, inplace=True)
    # a.drop('Q6', axis=1, inplace=True)
    print("================== CLEANED MATRIX ===============")
    print(a)
    print("---"*20)
    ## Chunk 1
    _nstudents_by2 = len(a)//2
    achunk1 = a.iloc[:_nstudents_by2].reset_index(drop=True).copy()
    achunk2 = a.iloc[_nstudents_by2:].reset_index(drop=True).copy()
    od = pd.DataFrame({'prof': profs})
    prof_sch = np.concatenate([
        get_schedule(achunk1, slots=slots//2),
        get_schedule(achunk2, slots=slots//2)+_nstudents_by2
    ], axis=1)
    res['assignment'] = (prof_sch>0).sum(axis=1)
    star_map = {1: "***", 2: "**", 3: "*", 4: "", 5: ""}
    students = a[ST_COL]
    students.loc[len(students)] = 'Not-Assigned'
    a.loc[len(a)] = {ST_COL: 'Not-Assigned'}
    a = a.fillna("")
    for s in range(slots):
        st_indices = prof_sch[:, s]
        st_indices[st_indices<0] = len(students)-1
        star_code = np.fromiter(
            map(lambda x: star_map.get(x, x), a.to_numpy()[st_indices, np.arange(len(prof_sch))+1]), dtype='U4'
        )
        od[f"slot{s}"] = students[st_indices].to_numpy() + star_code

    print("=="*30)
    print("IS the Mapping fair:")
    print(res)
    print("=="*30)
    od.to_csv(f"schedule_{form_csv_file}", index=False)
    return a

## d is a pandas dataframe rows are students and columns are professors
def get_schedule(d, slots=4):
    nstudent, nprof = d.shape[0], d.shape[1] - 1
    prof_sch = np.zeros((nprof, slots), dtype=int) - 9999
    st_meet_cnt = np.zeros(nstudent, dtype=int)
    approx_match_per_student = min(int(nprof * slots / nstudent + 0.5), slots)
    approx_match_per_faculty = min(int(nstudent * approx_match_per_student / nprof-0.5), slots)
    print(f"Each student should meet around: {approx_match_per_student} profs,"
          f"and Each prof should meet around {approx_match_per_faculty} students."
          f"nstudent={nstudent}, nprof={nprof}, slots={slots}")
    # Preference matrix
    a = d.to_numpy()[:, 1:].astype(np.float32)
    for i in range(slots):
        row_inds, col_inds = linear_sum_assignment(a)
        a[row_inds, col_inds] = 999
        print(f"Match-{i+1}: {list(zip(row_inds.tolist(), d[ST_COL][row_inds], d.columns[col_inds+1]))}")
        st_meet_cnt[row_inds] += 1
        prof_sch[col_inds, i] = row_inds
        _ismeeting_toomany, = np.where(st_meet_cnt>approx_match_per_student)
        _istooempty_prof, = np.where((prof_sch<0).sum(axis=1) > 1)
        if len(_ismeeting_toomany)>0:
            a[_ismeeting_toomany, :] = 999
            print(f"These students are meeting more than {approx_match_per_student} profs:")
            print(f"{d[ST_COL][_ismeeting_toomany]}")
        if len(_istooempty_prof)>0:
            a[:, _istooempty_prof] -= 0.75
            print(f"These profs are not getting matched, meeting less than {approx_match_per_faculty} students:")
            print(f"{d.columns[_istooempty_prof+1]}")

    print(f"Student meet counts: {st_meet_cnt}, Prof. meet counts: (prof_sch>=0).sum(axis=1)")

    ## Ensure the matching is accurate
    assert (st_meet_cnt>0).all(), "Some students are not meeting any professors"
    assert (st_meet_cnt<=(approx_match_per_student+1)).all(), "Some students are meeting too many professors"
    verify(prof_sch)
    return prof_sch

def verify(prof_sch):
    """
    matches is a dictionary from a student to a list of prof
    """
    nprof, slots = prof_sch.shape
    nstudents = prof_sch.max()+1
    print("=="*30)
    print(f"Schedule for {nstudents} students among {nprof} professors in {slots} slots.")
    print("=="*30)
    for i in range(nprof):
        s = prof_sch[i][prof_sch[i]>-1]        
        np.testing.assert_equal(len(np.unique(s)), len(s), f"Prof {i} meeting same student twice {s}.")
    for s in range(slots):
        st_involved = prof_sch[:, s][prof_sch[:, s] > 0]
        np.testing.assert_equal(len(np.unique(st_involved)), len(st_involved),
                                f"Some students are meeting multuiple faculty in the same slot {s}: {st_involved}.")

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
