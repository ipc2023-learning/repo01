begin_version
3
end_version
begin_metric
0
end_metric
14
begin_variable
var0
-1
2
Atom power_on(instrument2)
NegatedAtom power_on(instrument2)
end_variable
begin_variable
var1
-1
2
Atom power_on(instrument3)
NegatedAtom power_on(instrument3)
end_variable
begin_variable
var2
-1
2
Atom power_on(instrument4)
NegatedAtom power_on(instrument4)
end_variable
begin_variable
var3
-1
2
Atom power_avail(satellite1)
NegatedAtom power_avail(satellite1)
end_variable
begin_variable
var4
-1
2
Atom power_on(instrument0)
NegatedAtom power_on(instrument0)
end_variable
begin_variable
var5
-1
2
Atom power_on(instrument1)
NegatedAtom power_on(instrument1)
end_variable
begin_variable
var6
-1
2
Atom power_avail(satellite0)
NegatedAtom power_avail(satellite0)
end_variable
begin_variable
var7
-1
11
Atom pointing(satellite1, groundstation4)
Atom pointing(satellite1, groundstation8)
Atom pointing(satellite1, groundstation9)
Atom pointing(satellite1, planet10)
Atom pointing(satellite1, star0)
Atom pointing(satellite1, star1)
Atom pointing(satellite1, star2)
Atom pointing(satellite1, star3)
Atom pointing(satellite1, star5)
Atom pointing(satellite1, star6)
Atom pointing(satellite1, star7)
end_variable
begin_variable
var8
-1
11
Atom pointing(satellite0, groundstation4)
Atom pointing(satellite0, groundstation8)
Atom pointing(satellite0, groundstation9)
Atom pointing(satellite0, planet10)
Atom pointing(satellite0, star0)
Atom pointing(satellite0, star1)
Atom pointing(satellite0, star2)
Atom pointing(satellite0, star3)
Atom pointing(satellite0, star5)
Atom pointing(satellite0, star6)
Atom pointing(satellite0, star7)
end_variable
begin_variable
var9
-1
2
Atom calibrated(instrument4)
NegatedAtom calibrated(instrument4)
end_variable
begin_variable
var10
-1
2
Atom calibrated(instrument3)
NegatedAtom calibrated(instrument3)
end_variable
begin_variable
var11
-1
2
Atom calibrated(instrument1)
NegatedAtom calibrated(instrument1)
end_variable
begin_variable
var12
-1
2
Atom calibrated(instrument0)
NegatedAtom calibrated(instrument0)
end_variable
begin_variable
var13
-1
2
Atom have_image(planet10, thermograph3)
NegatedAtom have_image(planet10, thermograph3)
end_variable
0
begin_state
1
1
1
0
1
1
0
4
8
1
1
1
1
1
end_state
begin_goal
3
7 0
8 7
13 0
end_goal
242
begin_operator
calibrate satellite0 instrument0 groundstation8
2
8 1
4 0
1
0 12 -1 0
1
end_operator
begin_operator
calibrate satellite0 instrument0 star0
2
8 4
4 0
1
0 12 -1 0
1
end_operator
begin_operator
calibrate satellite0 instrument0 star7
2
8 10
4 0
1
0 12 -1 0
1
end_operator
begin_operator
calibrate satellite0 instrument1 groundstation8
2
8 1
5 0
1
0 11 -1 0
1
end_operator
begin_operator
calibrate satellite0 instrument1 groundstation9
2
8 2
5 0
1
0 11 -1 0
1
end_operator
begin_operator
calibrate satellite0 instrument1 star7
2
8 10
5 0
1
0 11 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument3 groundstation9
2
7 2
1 0
1
0 10 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument4 star7
2
7 10
2 0
1
0 9 -1 0
1
end_operator
begin_operator
switch_off instrument0 satellite0
0
2
0 6 -1 0
0 4 0 1
1
end_operator
begin_operator
switch_off instrument1 satellite0
0
2
0 6 -1 0
0 5 0 1
1
end_operator
begin_operator
switch_off instrument2 satellite1
0
2
0 3 -1 0
0 0 0 1
1
end_operator
begin_operator
switch_off instrument3 satellite1
0
2
0 3 -1 0
0 1 0 1
1
end_operator
begin_operator
switch_off instrument4 satellite1
0
2
0 3 -1 0
0 2 0 1
1
end_operator
begin_operator
switch_on instrument0 satellite0
0
3
0 12 -1 1
0 6 0 1
0 4 -1 0
1
end_operator
begin_operator
switch_on instrument1 satellite0
0
3
0 11 -1 1
0 6 0 1
0 5 -1 0
1
end_operator
begin_operator
switch_on instrument2 satellite1
0
2
0 3 0 1
0 0 -1 0
1
end_operator
begin_operator
switch_on instrument3 satellite1
0
3
0 10 -1 1
0 3 0 1
0 1 -1 0
1
end_operator
begin_operator
switch_on instrument4 satellite1
0
3
0 9 -1 1
0 3 0 1
0 2 -1 0
1
end_operator
begin_operator
take_image satellite0 planet10 instrument0 thermograph3
3
12 0
8 3
4 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite0 planet10 instrument1 thermograph3
3
11 0
8 3
5 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite1 planet10 instrument3 thermograph3
3
10 0
7 3
1 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite1 planet10 instrument4 thermograph3
3
9 0
7 3
2 0
1
0 13 -1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation8
0
1
0 8 1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation9
0
1
0 8 2 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 planet10
0
1
0 8 3 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star0
0
1
0 8 4 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star1
0
1
0 8 5 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star2
0
1
0 8 6 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star3
0
1
0 8 7 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star5
0
1
0 8 8 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star6
0
1
0 8 9 0
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star7
0
1
0 8 10 0
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation4
0
1
0 8 0 1
1
end_operator
begin_operator
turn_to satellite0 groundstation8 groundstation9
0
1
0 8 2 1
1
end_operator
begin_operator
turn_to satellite0 groundstation8 planet10
0
1
0 8 3 1
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star0
0
1
0 8 4 1
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star1
0
1
0 8 5 1
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star2
0
1
0 8 6 1
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star3
0
1
0 8 7 1
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star5
0
1
0 8 8 1
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star6
0
1
0 8 9 1
1
end_operator
begin_operator
turn_to satellite0 groundstation8 star7
0
1
0 8 10 1
1
end_operator
begin_operator
turn_to satellite0 groundstation9 groundstation4
0
1
0 8 0 2
1
end_operator
begin_operator
turn_to satellite0 groundstation9 groundstation8
0
1
0 8 1 2
1
end_operator
begin_operator
turn_to satellite0 groundstation9 planet10
0
1
0 8 3 2
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star0
0
1
0 8 4 2
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star1
0
1
0 8 5 2
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star2
0
1
0 8 6 2
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star3
0
1
0 8 7 2
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star5
0
1
0 8 8 2
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star6
0
1
0 8 9 2
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star7
0
1
0 8 10 2
1
end_operator
begin_operator
turn_to satellite0 planet10 groundstation4
0
1
0 8 0 3
1
end_operator
begin_operator
turn_to satellite0 planet10 groundstation8
0
1
0 8 1 3
1
end_operator
begin_operator
turn_to satellite0 planet10 groundstation9
0
1
0 8 2 3
1
end_operator
begin_operator
turn_to satellite0 planet10 star0
0
1
0 8 4 3
1
end_operator
begin_operator
turn_to satellite0 planet10 star1
0
1
0 8 5 3
1
end_operator
begin_operator
turn_to satellite0 planet10 star2
0
1
0 8 6 3
1
end_operator
begin_operator
turn_to satellite0 planet10 star3
0
1
0 8 7 3
1
end_operator
begin_operator
turn_to satellite0 planet10 star5
0
1
0 8 8 3
1
end_operator
begin_operator
turn_to satellite0 planet10 star6
0
1
0 8 9 3
1
end_operator
begin_operator
turn_to satellite0 planet10 star7
0
1
0 8 10 3
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation4
0
1
0 8 0 4
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation8
0
1
0 8 1 4
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation9
0
1
0 8 2 4
1
end_operator
begin_operator
turn_to satellite0 star0 planet10
0
1
0 8 3 4
1
end_operator
begin_operator
turn_to satellite0 star0 star1
0
1
0 8 5 4
1
end_operator
begin_operator
turn_to satellite0 star0 star2
0
1
0 8 6 4
1
end_operator
begin_operator
turn_to satellite0 star0 star3
0
1
0 8 7 4
1
end_operator
begin_operator
turn_to satellite0 star0 star5
0
1
0 8 8 4
1
end_operator
begin_operator
turn_to satellite0 star0 star6
0
1
0 8 9 4
1
end_operator
begin_operator
turn_to satellite0 star0 star7
0
1
0 8 10 4
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation4
0
1
0 8 0 5
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation8
0
1
0 8 1 5
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation9
0
1
0 8 2 5
1
end_operator
begin_operator
turn_to satellite0 star1 planet10
0
1
0 8 3 5
1
end_operator
begin_operator
turn_to satellite0 star1 star0
0
1
0 8 4 5
1
end_operator
begin_operator
turn_to satellite0 star1 star2
0
1
0 8 6 5
1
end_operator
begin_operator
turn_to satellite0 star1 star3
0
1
0 8 7 5
1
end_operator
begin_operator
turn_to satellite0 star1 star5
0
1
0 8 8 5
1
end_operator
begin_operator
turn_to satellite0 star1 star6
0
1
0 8 9 5
1
end_operator
begin_operator
turn_to satellite0 star1 star7
0
1
0 8 10 5
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation4
0
1
0 8 0 6
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation8
0
1
0 8 1 6
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation9
0
1
0 8 2 6
1
end_operator
begin_operator
turn_to satellite0 star2 planet10
0
1
0 8 3 6
1
end_operator
begin_operator
turn_to satellite0 star2 star0
0
1
0 8 4 6
1
end_operator
begin_operator
turn_to satellite0 star2 star1
0
1
0 8 5 6
1
end_operator
begin_operator
turn_to satellite0 star2 star3
0
1
0 8 7 6
1
end_operator
begin_operator
turn_to satellite0 star2 star5
0
1
0 8 8 6
1
end_operator
begin_operator
turn_to satellite0 star2 star6
0
1
0 8 9 6
1
end_operator
begin_operator
turn_to satellite0 star2 star7
0
1
0 8 10 6
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation4
0
1
0 8 0 7
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation8
0
1
0 8 1 7
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation9
0
1
0 8 2 7
1
end_operator
begin_operator
turn_to satellite0 star3 planet10
0
1
0 8 3 7
1
end_operator
begin_operator
turn_to satellite0 star3 star0
0
1
0 8 4 7
1
end_operator
begin_operator
turn_to satellite0 star3 star1
0
1
0 8 5 7
1
end_operator
begin_operator
turn_to satellite0 star3 star2
0
1
0 8 6 7
1
end_operator
begin_operator
turn_to satellite0 star3 star5
0
1
0 8 8 7
1
end_operator
begin_operator
turn_to satellite0 star3 star6
0
1
0 8 9 7
1
end_operator
begin_operator
turn_to satellite0 star3 star7
0
1
0 8 10 7
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation4
0
1
0 8 0 8
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation8
0
1
0 8 1 8
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation9
0
1
0 8 2 8
1
end_operator
begin_operator
turn_to satellite0 star5 planet10
0
1
0 8 3 8
1
end_operator
begin_operator
turn_to satellite0 star5 star0
0
1
0 8 4 8
1
end_operator
begin_operator
turn_to satellite0 star5 star1
0
1
0 8 5 8
1
end_operator
begin_operator
turn_to satellite0 star5 star2
0
1
0 8 6 8
1
end_operator
begin_operator
turn_to satellite0 star5 star3
0
1
0 8 7 8
1
end_operator
begin_operator
turn_to satellite0 star5 star6
0
1
0 8 9 8
1
end_operator
begin_operator
turn_to satellite0 star5 star7
0
1
0 8 10 8
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation4
0
1
0 8 0 9
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation8
0
1
0 8 1 9
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation9
0
1
0 8 2 9
1
end_operator
begin_operator
turn_to satellite0 star6 planet10
0
1
0 8 3 9
1
end_operator
begin_operator
turn_to satellite0 star6 star0
0
1
0 8 4 9
1
end_operator
begin_operator
turn_to satellite0 star6 star1
0
1
0 8 5 9
1
end_operator
begin_operator
turn_to satellite0 star6 star2
0
1
0 8 6 9
1
end_operator
begin_operator
turn_to satellite0 star6 star3
0
1
0 8 7 9
1
end_operator
begin_operator
turn_to satellite0 star6 star5
0
1
0 8 8 9
1
end_operator
begin_operator
turn_to satellite0 star6 star7
0
1
0 8 10 9
1
end_operator
begin_operator
turn_to satellite0 star7 groundstation4
0
1
0 8 0 10
1
end_operator
begin_operator
turn_to satellite0 star7 groundstation8
0
1
0 8 1 10
1
end_operator
begin_operator
turn_to satellite0 star7 groundstation9
0
1
0 8 2 10
1
end_operator
begin_operator
turn_to satellite0 star7 planet10
0
1
0 8 3 10
1
end_operator
begin_operator
turn_to satellite0 star7 star0
0
1
0 8 4 10
1
end_operator
begin_operator
turn_to satellite0 star7 star1
0
1
0 8 5 10
1
end_operator
begin_operator
turn_to satellite0 star7 star2
0
1
0 8 6 10
1
end_operator
begin_operator
turn_to satellite0 star7 star3
0
1
0 8 7 10
1
end_operator
begin_operator
turn_to satellite0 star7 star5
0
1
0 8 8 10
1
end_operator
begin_operator
turn_to satellite0 star7 star6
0
1
0 8 9 10
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation8
0
1
0 7 1 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation9
0
1
0 7 2 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 planet10
0
1
0 7 3 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star0
0
1
0 7 4 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star1
0
1
0 7 5 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star2
0
1
0 7 6 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star3
0
1
0 7 7 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star5
0
1
0 7 8 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star6
0
1
0 7 9 0
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star7
0
1
0 7 10 0
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation4
0
1
0 7 0 1
1
end_operator
begin_operator
turn_to satellite1 groundstation8 groundstation9
0
1
0 7 2 1
1
end_operator
begin_operator
turn_to satellite1 groundstation8 planet10
0
1
0 7 3 1
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star0
0
1
0 7 4 1
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star1
0
1
0 7 5 1
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star2
0
1
0 7 6 1
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star3
0
1
0 7 7 1
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star5
0
1
0 7 8 1
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star6
0
1
0 7 9 1
1
end_operator
begin_operator
turn_to satellite1 groundstation8 star7
0
1
0 7 10 1
1
end_operator
begin_operator
turn_to satellite1 groundstation9 groundstation4
0
1
0 7 0 2
1
end_operator
begin_operator
turn_to satellite1 groundstation9 groundstation8
0
1
0 7 1 2
1
end_operator
begin_operator
turn_to satellite1 groundstation9 planet10
0
1
0 7 3 2
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star0
0
1
0 7 4 2
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star1
0
1
0 7 5 2
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star2
0
1
0 7 6 2
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star3
0
1
0 7 7 2
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star5
0
1
0 7 8 2
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star6
0
1
0 7 9 2
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star7
0
1
0 7 10 2
1
end_operator
begin_operator
turn_to satellite1 planet10 groundstation4
0
1
0 7 0 3
1
end_operator
begin_operator
turn_to satellite1 planet10 groundstation8
0
1
0 7 1 3
1
end_operator
begin_operator
turn_to satellite1 planet10 groundstation9
0
1
0 7 2 3
1
end_operator
begin_operator
turn_to satellite1 planet10 star0
0
1
0 7 4 3
1
end_operator
begin_operator
turn_to satellite1 planet10 star1
0
1
0 7 5 3
1
end_operator
begin_operator
turn_to satellite1 planet10 star2
0
1
0 7 6 3
1
end_operator
begin_operator
turn_to satellite1 planet10 star3
0
1
0 7 7 3
1
end_operator
begin_operator
turn_to satellite1 planet10 star5
0
1
0 7 8 3
1
end_operator
begin_operator
turn_to satellite1 planet10 star6
0
1
0 7 9 3
1
end_operator
begin_operator
turn_to satellite1 planet10 star7
0
1
0 7 10 3
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation4
0
1
0 7 0 4
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation8
0
1
0 7 1 4
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation9
0
1
0 7 2 4
1
end_operator
begin_operator
turn_to satellite1 star0 planet10
0
1
0 7 3 4
1
end_operator
begin_operator
turn_to satellite1 star0 star1
0
1
0 7 5 4
1
end_operator
begin_operator
turn_to satellite1 star0 star2
0
1
0 7 6 4
1
end_operator
begin_operator
turn_to satellite1 star0 star3
0
1
0 7 7 4
1
end_operator
begin_operator
turn_to satellite1 star0 star5
0
1
0 7 8 4
1
end_operator
begin_operator
turn_to satellite1 star0 star6
0
1
0 7 9 4
1
end_operator
begin_operator
turn_to satellite1 star0 star7
0
1
0 7 10 4
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation4
0
1
0 7 0 5
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation8
0
1
0 7 1 5
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation9
0
1
0 7 2 5
1
end_operator
begin_operator
turn_to satellite1 star1 planet10
0
1
0 7 3 5
1
end_operator
begin_operator
turn_to satellite1 star1 star0
0
1
0 7 4 5
1
end_operator
begin_operator
turn_to satellite1 star1 star2
0
1
0 7 6 5
1
end_operator
begin_operator
turn_to satellite1 star1 star3
0
1
0 7 7 5
1
end_operator
begin_operator
turn_to satellite1 star1 star5
0
1
0 7 8 5
1
end_operator
begin_operator
turn_to satellite1 star1 star6
0
1
0 7 9 5
1
end_operator
begin_operator
turn_to satellite1 star1 star7
0
1
0 7 10 5
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation4
0
1
0 7 0 6
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation8
0
1
0 7 1 6
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation9
0
1
0 7 2 6
1
end_operator
begin_operator
turn_to satellite1 star2 planet10
0
1
0 7 3 6
1
end_operator
begin_operator
turn_to satellite1 star2 star0
0
1
0 7 4 6
1
end_operator
begin_operator
turn_to satellite1 star2 star1
0
1
0 7 5 6
1
end_operator
begin_operator
turn_to satellite1 star2 star3
0
1
0 7 7 6
1
end_operator
begin_operator
turn_to satellite1 star2 star5
0
1
0 7 8 6
1
end_operator
begin_operator
turn_to satellite1 star2 star6
0
1
0 7 9 6
1
end_operator
begin_operator
turn_to satellite1 star2 star7
0
1
0 7 10 6
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation4
0
1
0 7 0 7
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation8
0
1
0 7 1 7
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation9
0
1
0 7 2 7
1
end_operator
begin_operator
turn_to satellite1 star3 planet10
0
1
0 7 3 7
1
end_operator
begin_operator
turn_to satellite1 star3 star0
0
1
0 7 4 7
1
end_operator
begin_operator
turn_to satellite1 star3 star1
0
1
0 7 5 7
1
end_operator
begin_operator
turn_to satellite1 star3 star2
0
1
0 7 6 7
1
end_operator
begin_operator
turn_to satellite1 star3 star5
0
1
0 7 8 7
1
end_operator
begin_operator
turn_to satellite1 star3 star6
0
1
0 7 9 7
1
end_operator
begin_operator
turn_to satellite1 star3 star7
0
1
0 7 10 7
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation4
0
1
0 7 0 8
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation8
0
1
0 7 1 8
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation9
0
1
0 7 2 8
1
end_operator
begin_operator
turn_to satellite1 star5 planet10
0
1
0 7 3 8
1
end_operator
begin_operator
turn_to satellite1 star5 star0
0
1
0 7 4 8
1
end_operator
begin_operator
turn_to satellite1 star5 star1
0
1
0 7 5 8
1
end_operator
begin_operator
turn_to satellite1 star5 star2
0
1
0 7 6 8
1
end_operator
begin_operator
turn_to satellite1 star5 star3
0
1
0 7 7 8
1
end_operator
begin_operator
turn_to satellite1 star5 star6
0
1
0 7 9 8
1
end_operator
begin_operator
turn_to satellite1 star5 star7
0
1
0 7 10 8
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation4
0
1
0 7 0 9
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation8
0
1
0 7 1 9
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation9
0
1
0 7 2 9
1
end_operator
begin_operator
turn_to satellite1 star6 planet10
0
1
0 7 3 9
1
end_operator
begin_operator
turn_to satellite1 star6 star0
0
1
0 7 4 9
1
end_operator
begin_operator
turn_to satellite1 star6 star1
0
1
0 7 5 9
1
end_operator
begin_operator
turn_to satellite1 star6 star2
0
1
0 7 6 9
1
end_operator
begin_operator
turn_to satellite1 star6 star3
0
1
0 7 7 9
1
end_operator
begin_operator
turn_to satellite1 star6 star5
0
1
0 7 8 9
1
end_operator
begin_operator
turn_to satellite1 star6 star7
0
1
0 7 10 9
1
end_operator
begin_operator
turn_to satellite1 star7 groundstation4
0
1
0 7 0 10
1
end_operator
begin_operator
turn_to satellite1 star7 groundstation8
0
1
0 7 1 10
1
end_operator
begin_operator
turn_to satellite1 star7 groundstation9
0
1
0 7 2 10
1
end_operator
begin_operator
turn_to satellite1 star7 planet10
0
1
0 7 3 10
1
end_operator
begin_operator
turn_to satellite1 star7 star0
0
1
0 7 4 10
1
end_operator
begin_operator
turn_to satellite1 star7 star1
0
1
0 7 5 10
1
end_operator
begin_operator
turn_to satellite1 star7 star2
0
1
0 7 6 10
1
end_operator
begin_operator
turn_to satellite1 star7 star3
0
1
0 7 7 10
1
end_operator
begin_operator
turn_to satellite1 star7 star5
0
1
0 7 8 10
1
end_operator
begin_operator
turn_to satellite1 star7 star6
0
1
0 7 9 10
1
end_operator
0
