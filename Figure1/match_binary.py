from ossos import parsers
from astropy.table import Table

binary = Table.read('binary_ordered.txt', format='ascii')
orbits = Table.read('OSSOSpp_fixed.txt', format='ascii')

# >>> parsers.mpcorb_desig_unpack('K13GD7O')

orbits['Target'] = [ parsers.mpcorb_desig_unpack(r['MPC']).replace(' ','') for r in orbits ]

for target in binary:
   orbit = orbits[orbits['Target']==target['Target']]
   if len(orbit) != 1:
      print(orbit)
      print(f"WARNING\n {target}")
      continue
   cc = (orbit['ifree'] < 4.2 ) & ( orbit['a'] > 42.4 ) & ( orbit['a'] < 47.4) & (orbit['cl']=='cla')
   cls = "CC" if cc else "HC"
   print(f"{cls} {target['Target']} {target['binary']} {target['HVsys']} {orbit['ifree'][0]} {orbit['i'][0]}")

