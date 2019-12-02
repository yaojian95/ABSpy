import unittest
import numpy as np
from abspy.tools.observable_dict import ObservableDict, Measurements, Masks

class TestObservableDicts(unittest.TestCase):
    
    def test_basedict(self):
        basedict = ObservableDict()
        self.assertEqual(basedict.archive, dict())
        
    def test_measuredict_append_array(self):
        measuredict = Measurements()
        hrr = np.random.rand(48)
        measuredict.append(('test', 'nan', '2', 'nan'), hrr)  # healpix array
        local_arr = measuredict[('test', 'nan', '2', 'nan')]
        self.assertListEqual(local_arr, list(hrr))
        
    def test_maskdict_append_array(self):
        msk = np.random.randint(0, 2, 48)
        mskdict = Masks()
        mskdict.append(('test', 'nan', '2', 'nan'), msk)
        local_msk = mskdict[('test', 'nan', '2', 'nan')]
        self.assertListEqual(local_msk, list(msk))
    
    def test_meadict_apply_mask(self):
        msk = np.random.randint(0, 2, 48)
        mskdict = Masks()
        mskdict.append(('test', 'nan', '2', 'nan'), msk)
        arr = np.random.rand(48)
        meadict = Measurements()
        meadict.append(('test', 'nan', '2', 'nan'), arr)
        meadict.apply_mask(mskdict)
        for i in range(len(arr)):
            arr[i] *= msk[i]
        self.assertListEqual(meadict[('test', 'nan', '2', 'nan')], list(arr))
        
        
if __name__ == '__main__':
    unittest.main()