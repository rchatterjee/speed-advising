import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


## d is a pandas dataframe rows are students and columns are professors
def get_schedule(d, slots=4):
    all_profs = d.columns[1:]
    nstudent, nprof = d.shape[0], d.shape[1] - 1
    prof_sch = np.zeros((nprof, slots), dtype=int) - 1
    st_meet_cnt = np.zeros(nstudent, dtype=int)
    
    approx_match_per_student = int(nprof * slots / nstudent)
    print(f"Each student should meet around: {approx_match_per_student} profs")
    # Preference matrix
    a = d.to_numpy()[:, 1:].astype(int)
    for i in range(slots):
        row_inds, col_inds = linear_sum_assignment(a)
        a[row_inds, col_inds] = 999
        print(f"Match-{i+1}: {list(zip(d['st'][row_inds], d.columns[col_inds+1]))}")
        st_meet_cnt[row_inds] += 1
        prof_sch[col_inds, i] = row_inds
        _ismeeting_toomany, = np.where(st_meet_cnt>approx_match_per_student)
        if len(_ismeeting_toomany)>0:
            a[_ismeeting_toomany, :] = 999
            print(_ismeeting_toomany)
            print(f"Adjusted some students' preferences. {d['st'][_ismeeting_toomany]}")
            print(a)
    print(f"Meet counts: {st_meet_cnt}")
    assert (st_meet_cnt>0).all(), "Some students are not meeting any professors"
    assert (st_meet_cnt<=(approx_match_per_student+1)).all(), "Some students are meeting too many professors"
    return prof_sch

def verify(d, prof_sch):
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
        sts = '\t'.join(d['st'][s])
        print(f"{p} => {sts}")
        np.testing.assert_equal(len(np.unique(s)), len(s), f"Prof {p} meeting same student twice {sts}.")
    for s in range(slots):
        st_involved = prof_sch[:, s][prof_sch[:, s] > 0]
        np.testing.assert_equal(len(np.unique(st_involved)), len(st_involved), f"Some students are meeting multuiple faculty in the same slot {s}: {st_involved}.")


if __name__ == "__main__":
    d = pd.read_csv('sp.csv')
    print(d)
    x = list(np.arange(d.shape[1]-1)+1)
    # np.random.shuffle(x)
    d = d.iloc[:, [0] + x]
    print(d)
    slots = 4
    matches = get_schedule(d, slots)
    verify(d, matches)
