from gates import Gates

def crossed(records):

    gates = None

    if not gates:
        _Gates = Gates()
        _Gates.load_gates()
        gates = _Gates.gates

    def ccw(a, b, c):
        return ((c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]))

    def cross(s1, s2):
        a, b = s1
        c, d = s2
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d) 


    # now lets cycle throught each objects trajectory and determine if it has crossed either of the gates
    def crossed_gates(records):
        #  compute detection centroids
        for record in records:
            if len(record) == 2:
                # get position at current time
                xy_t0 = tuple(record[1][-1:][0])
                # get position at most recent historic time step
                xy_t1 = tuple(record[0][-1:][0])


                # cycle through gates
                for g, gate in enumerate(gates):
                    if cross(gates[g], [xy_t0, xy_t1]):
                        print(f'crossed - {g} @ {record[0]}')
                        '''
                        timecat.insert(0, g)
                        timecat.insert(0, DEVICE.UUID)
                        detections.insert(0, timecat)
                        '''

    return(crossed_gates(records))

print(crossed('test'))