import datetime
import os
import unittest

import cf

class aggregateTest(unittest.TestCase):
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'test_file.nc')
    chunk_sizes = (17, 34, 300, 100000)[::-1]
    original_chunksize = cf.CHUNKSIZE()
    

    def test_aggregate(self):
        for chunksize in self.chunk_sizes:    
            cf.CHUNKSIZE(chunksize)

            f = cf.read(self.filename, squeeze=True)[0]
       
            g = cf.FieldList(f[0])
            g.append(f[1:3])
            g.append(f[3])
            g[-1].flip(0, inplace=True)
            g.append(f[4:7])
            g[-1].flip(0, inplace=True)
            g.extend([f[i] for i in range(7, f.shape[0])])

            g0 = g.copy()
            self.assertTrue(g.equals(g0, verbose=True), "g != g0")
            
            h = cf.aggregate(g, info=1)

            self.assertTrue(len(h) == 1)

            self.assertTrue(h[0].shape == (10, 9), 'h[0].shape = '+repr(h[0].shape)+' != (10, 9)')
        
            self.assertTrue(g.equals(g0, verbose=True), 'g != itself after aggregation')

            self.assertTrue(h[0].equals(f, verbose=True), 'h[0] != f')
            
            i = cf.aggregate(g, info=1)
            
            self.assertTrue(i.equals(h, verbose=True), 'The second aggregation != the first')
            
            self.assertTrue(g.equals(g0, verbose=True), 'g != itself after the second aggregation')
            
            i = cf.aggregate(g, info=1, axes='grid_latitude')
            
            self.assertTrue(i.equals(h, verbose=True), 'The third aggregation != the first')
            
            self.assertTrue(g.equals(g0, verbose=True), 'g !=itself after the third aggregation')
            
            self.assertTrue(i[0].shape == (10,9), 'i[0].shape is '+repr(i[0].shape))
            
            i = cf.aggregate(g, info=1, axes='grid_latitude', donotchecknonaggregatingaxes=1)
            
            self.assertTrue(i.equals(h, verbose=True), 'The fourth aggregation != the first')
            
            self.assertTrue(g.equals(g0, verbose=True), 'g != itself after the fourth aggregation')
            
            self.assertTrue(i[0].shape == (10,9), 'i[0].shape is '+repr(i[0].shape))

            #
            q, t = cf.read('file.nc')
            c = cf.read('file2.nc')[0]

            d = cf.aggregate([c, t], info=0, relaxed_identities=True)
            e = cf.aggregate([t, c], info=0, relaxed_identities=True)

            self.assertTrue(len(d) == 1)
            self.assertTrue(len(e) == 1)
            self.assertTrue(d[0].shape == (3,) + t.shape)           
            self.assertTrue(d[0].equals(e[0], verbose=True))

            x = cf.read(['file.nc', 'file2.nc'], aggregate=False)
            self.assertTrue(len(x) == 3)

            x = cf.read(['file.nc', 'file2.nc'], aggregate={'relaxed_identities': True})
            self.assertTrue(len(x) == 2)

        cf.CHUNKSIZE(self.original_chunksize)


#--- End: class

if __name__ == "__main__":
    print('Run date:', datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
