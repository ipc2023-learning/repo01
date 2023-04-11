begin_version
3
end_version
begin_metric
0
end_metric
16
begin_variable
var0
-1
2
Atom power_on(instrument5)
NegatedAtom power_on(instrument5)
end_variable
begin_variable
var1
-1
2
Atom power_on(instrument6)
NegatedAtom power_on(instrument6)
end_variable
begin_variable
var2
-1
2
Atom power_avail(satellite3)
NegatedAtom power_avail(satellite3)
end_variable
begin_variable
var3
-1
2
Atom power_avail(satellite2)
NegatedAtom power_avail(satellite2)
end_variable
begin_variable
var4
-1
2
Atom power_on(instrument4)
NegatedAtom power_on(instrument4)
end_variable
begin_variable
var5
-1
2
Atom power_on(instrument0)
NegatedAtom power_on(instrument0)
end_variable
begin_variable
var6
-1
2
Atom power_on(instrument1)
NegatedAtom power_on(instrument1)
end_variable
begin_variable
var7
-1
2
Atom power_avail(satellite0)
NegatedAtom power_avail(satellite0)
end_variable
begin_variable
var8
-1
11
Atom pointing(satellite3, groundstation1)
Atom pointing(satellite3, groundstation2)
Atom pointing(satellite3, groundstation3)
Atom pointing(satellite3, groundstation4)
Atom pointing(satellite3, groundstation7)
Atom pointing(satellite3, planet10)
Atom pointing(satellite3, star0)
Atom pointing(satellite3, star5)
Atom pointing(satellite3, star6)
Atom pointing(satellite3, star8)
Atom pointing(satellite3, star9)
end_variable
begin_variable
var9
-1
11
Atom pointing(satellite2, groundstation1)
Atom pointing(satellite2, groundstation2)
Atom pointing(satellite2, groundstation3)
Atom pointing(satellite2, groundstation4)
Atom pointing(satellite2, groundstation7)
Atom pointing(satellite2, planet10)
Atom pointing(satellite2, star0)
Atom pointing(satellite2, star5)
Atom pointing(satellite2, star6)
Atom pointing(satellite2, star8)
Atom pointing(satellite2, star9)
end_variable
begin_variable
var10
-1
11
Atom pointing(satellite1, groundstation1)
Atom pointing(satellite1, groundstation2)
Atom pointing(satellite1, groundstation3)
Atom pointing(satellite1, groundstation4)
Atom pointing(satellite1, groundstation7)
Atom pointing(satellite1, planet10)
Atom pointing(satellite1, star0)
Atom pointing(satellite1, star5)
Atom pointing(satellite1, star6)
Atom pointing(satellite1, star8)
Atom pointing(satellite1, star9)
end_variable
begin_variable
var11
-1
11
Atom pointing(satellite0, groundstation1)
Atom pointing(satellite0, groundstation2)
Atom pointing(satellite0, groundstation3)
Atom pointing(satellite0, groundstation4)
Atom pointing(satellite0, groundstation7)
Atom pointing(satellite0, planet10)
Atom pointing(satellite0, star0)
Atom pointing(satellite0, star5)
Atom pointing(satellite0, star6)
Atom pointing(satellite0, star8)
Atom pointing(satellite0, star9)
end_variable
begin_variable
var12
-1
2
Atom calibrated(instrument5)
NegatedAtom calibrated(instrument5)
end_variable
begin_variable
var13
-1
2
Atom calibrated(instrument4)
NegatedAtom calibrated(instrument4)
end_variable
begin_variable
var14
-1
2
Atom calibrated(instrument1)
NegatedAtom calibrated(instrument1)
end_variable
begin_variable
var15
-1
2
Atom have_image(planet10, thermograph2)
NegatedAtom have_image(planet10, thermograph2)
end_variable
0
begin_state
1
1
0
0
1
1
1
0
4
1
9
9
1
1
1
1
end_state
begin_goal
2
10 10
15 0
end_goal
460
begin_operator
calibrate satellite0 instrument1 groundstation1
2
11 0
6 0
1
0 14 -1 0
1
end_operator
begin_operator
calibrate satellite0 instrument1 star5
2
11 7
6 0
1
0 14 -1 0
1
end_operator
begin_operator
calibrate satellite0 instrument1 star9
2
11 10
6 0
1
0 14 -1 0
1
end_operator
begin_operator
calibrate satellite2 instrument4 groundstation2
2
9 1
4 0
1
0 13 -1 0
1
end_operator
begin_operator
calibrate satellite2 instrument4 groundstation7
2
9 4
4 0
1
0 13 -1 0
1
end_operator
begin_operator
calibrate satellite3 instrument5 groundstation7
2
8 4
0 0
1
0 12 -1 0
1
end_operator
begin_operator
calibrate satellite3 instrument5 star9
2
8 10
0 0
1
0 12 -1 0
1
end_operator
begin_operator
switch_off instrument0 satellite0
0
2
0 7 -1 0
0 5 0 1
1
end_operator
begin_operator
switch_off instrument1 satellite0
0
2
0 7 -1 0
0 6 0 1
1
end_operator
begin_operator
switch_off instrument4 satellite2
0
2
0 3 -1 0
0 4 0 1
1
end_operator
begin_operator
switch_off instrument5 satellite3
0
2
0 2 -1 0
0 0 0 1
1
end_operator
begin_operator
switch_off instrument6 satellite3
0
2
0 2 -1 0
0 1 0 1
1
end_operator
begin_operator
switch_on instrument0 satellite0
0
2
0 7 0 1
0 5 -1 0
1
end_operator
begin_operator
switch_on instrument1 satellite0
0
3
0 14 -1 1
0 7 0 1
0 6 -1 0
1
end_operator
begin_operator
switch_on instrument4 satellite2
0
3
0 13 -1 1
0 3 0 1
0 4 -1 0
1
end_operator
begin_operator
switch_on instrument5 satellite3
0
3
0 12 -1 1
0 2 0 1
0 0 -1 0
1
end_operator
begin_operator
switch_on instrument6 satellite3
0
2
0 2 0 1
0 1 -1 0
1
end_operator
begin_operator
take_image satellite0 planet10 instrument1 thermograph2
3
14 0
11 5
6 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite2 planet10 instrument4 thermograph2
3
13 0
9 5
4 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite3 planet10 instrument5 thermograph2
3
12 0
8 5
0 0
1
0 15 -1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation2
0
1
0 11 1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation3
0
1
0 11 2 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation4
0
1
0 11 3 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation7
0
1
0 11 4 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 planet10
0
1
0 11 5 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star0
0
1
0 11 6 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star5
0
1
0 11 7 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star6
0
1
0 11 8 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star8
0
1
0 11 9 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star9
0
1
0 11 10 0
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation1
0
1
0 11 0 1
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation3
0
1
0 11 2 1
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation4
0
1
0 11 3 1
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation7
0
1
0 11 4 1
1
end_operator
begin_operator
turn_to satellite0 groundstation2 planet10
0
1
0 11 5 1
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star0
0
1
0 11 6 1
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star5
0
1
0 11 7 1
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star6
0
1
0 11 8 1
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star8
0
1
0 11 9 1
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star9
0
1
0 11 10 1
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation1
0
1
0 11 0 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation2
0
1
0 11 1 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation4
0
1
0 11 3 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation7
0
1
0 11 4 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 planet10
0
1
0 11 5 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star0
0
1
0 11 6 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star5
0
1
0 11 7 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star6
0
1
0 11 8 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star8
0
1
0 11 9 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star9
0
1
0 11 10 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation1
0
1
0 11 0 3
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation2
0
1
0 11 1 3
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation3
0
1
0 11 2 3
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation7
0
1
0 11 4 3
1
end_operator
begin_operator
turn_to satellite0 groundstation4 planet10
0
1
0 11 5 3
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star0
0
1
0 11 6 3
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star5
0
1
0 11 7 3
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star6
0
1
0 11 8 3
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star8
0
1
0 11 9 3
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star9
0
1
0 11 10 3
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation1
0
1
0 11 0 4
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation2
0
1
0 11 1 4
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation3
0
1
0 11 2 4
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation4
0
1
0 11 3 4
1
end_operator
begin_operator
turn_to satellite0 groundstation7 planet10
0
1
0 11 5 4
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star0
0
1
0 11 6 4
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star5
0
1
0 11 7 4
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star6
0
1
0 11 8 4
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star8
0
1
0 11 9 4
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star9
0
1
0 11 10 4
1
end_operator
begin_operator
turn_to satellite0 planet10 groundstation1
0
1
0 11 0 5
1
end_operator
begin_operator
turn_to satellite0 planet10 groundstation2
0
1
0 11 1 5
1
end_operator
begin_operator
turn_to satellite0 planet10 groundstation3
0
1
0 11 2 5
1
end_operator
begin_operator
turn_to satellite0 planet10 groundstation4
0
1
0 11 3 5
1
end_operator
begin_operator
turn_to satellite0 planet10 groundstation7
0
1
0 11 4 5
1
end_operator
begin_operator
turn_to satellite0 planet10 star0
0
1
0 11 6 5
1
end_operator
begin_operator
turn_to satellite0 planet10 star5
0
1
0 11 7 5
1
end_operator
begin_operator
turn_to satellite0 planet10 star6
0
1
0 11 8 5
1
end_operator
begin_operator
turn_to satellite0 planet10 star8
0
1
0 11 9 5
1
end_operator
begin_operator
turn_to satellite0 planet10 star9
0
1
0 11 10 5
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation1
0
1
0 11 0 6
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation2
0
1
0 11 1 6
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation3
0
1
0 11 2 6
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation4
0
1
0 11 3 6
1
end_operator
begin_operator
turn_to satellite0 star0 groundstation7
0
1
0 11 4 6
1
end_operator
begin_operator
turn_to satellite0 star0 planet10
0
1
0 11 5 6
1
end_operator
begin_operator
turn_to satellite0 star0 star5
0
1
0 11 7 6
1
end_operator
begin_operator
turn_to satellite0 star0 star6
0
1
0 11 8 6
1
end_operator
begin_operator
turn_to satellite0 star0 star8
0
1
0 11 9 6
1
end_operator
begin_operator
turn_to satellite0 star0 star9
0
1
0 11 10 6
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation1
0
1
0 11 0 7
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation2
0
1
0 11 1 7
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation3
0
1
0 11 2 7
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation4
0
1
0 11 3 7
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation7
0
1
0 11 4 7
1
end_operator
begin_operator
turn_to satellite0 star5 planet10
0
1
0 11 5 7
1
end_operator
begin_operator
turn_to satellite0 star5 star0
0
1
0 11 6 7
1
end_operator
begin_operator
turn_to satellite0 star5 star6
0
1
0 11 8 7
1
end_operator
begin_operator
turn_to satellite0 star5 star8
0
1
0 11 9 7
1
end_operator
begin_operator
turn_to satellite0 star5 star9
0
1
0 11 10 7
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation1
0
1
0 11 0 8
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation2
0
1
0 11 1 8
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation3
0
1
0 11 2 8
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation4
0
1
0 11 3 8
1
end_operator
begin_operator
turn_to satellite0 star6 groundstation7
0
1
0 11 4 8
1
end_operator
begin_operator
turn_to satellite0 star6 planet10
0
1
0 11 5 8
1
end_operator
begin_operator
turn_to satellite0 star6 star0
0
1
0 11 6 8
1
end_operator
begin_operator
turn_to satellite0 star6 star5
0
1
0 11 7 8
1
end_operator
begin_operator
turn_to satellite0 star6 star8
0
1
0 11 9 8
1
end_operator
begin_operator
turn_to satellite0 star6 star9
0
1
0 11 10 8
1
end_operator
begin_operator
turn_to satellite0 star8 groundstation1
0
1
0 11 0 9
1
end_operator
begin_operator
turn_to satellite0 star8 groundstation2
0
1
0 11 1 9
1
end_operator
begin_operator
turn_to satellite0 star8 groundstation3
0
1
0 11 2 9
1
end_operator
begin_operator
turn_to satellite0 star8 groundstation4
0
1
0 11 3 9
1
end_operator
begin_operator
turn_to satellite0 star8 groundstation7
0
1
0 11 4 9
1
end_operator
begin_operator
turn_to satellite0 star8 planet10
0
1
0 11 5 9
1
end_operator
begin_operator
turn_to satellite0 star8 star0
0
1
0 11 6 9
1
end_operator
begin_operator
turn_to satellite0 star8 star5
0
1
0 11 7 9
1
end_operator
begin_operator
turn_to satellite0 star8 star6
0
1
0 11 8 9
1
end_operator
begin_operator
turn_to satellite0 star8 star9
0
1
0 11 10 9
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation1
0
1
0 11 0 10
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation2
0
1
0 11 1 10
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation3
0
1
0 11 2 10
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation4
0
1
0 11 3 10
1
end_operator
begin_operator
turn_to satellite0 star9 groundstation7
0
1
0 11 4 10
1
end_operator
begin_operator
turn_to satellite0 star9 planet10
0
1
0 11 5 10
1
end_operator
begin_operator
turn_to satellite0 star9 star0
0
1
0 11 6 10
1
end_operator
begin_operator
turn_to satellite0 star9 star5
0
1
0 11 7 10
1
end_operator
begin_operator
turn_to satellite0 star9 star6
0
1
0 11 8 10
1
end_operator
begin_operator
turn_to satellite0 star9 star8
0
1
0 11 9 10
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation2
0
1
0 10 1 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation3
0
1
0 10 2 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation4
0
1
0 10 3 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation7
0
1
0 10 4 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 planet10
0
1
0 10 5 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star0
0
1
0 10 6 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star5
0
1
0 10 7 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star6
0
1
0 10 8 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star8
0
1
0 10 9 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star9
0
1
0 10 10 0
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation1
0
1
0 10 0 1
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation3
0
1
0 10 2 1
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation4
0
1
0 10 3 1
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation7
0
1
0 10 4 1
1
end_operator
begin_operator
turn_to satellite1 groundstation2 planet10
0
1
0 10 5 1
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star0
0
1
0 10 6 1
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star5
0
1
0 10 7 1
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star6
0
1
0 10 8 1
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star8
0
1
0 10 9 1
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star9
0
1
0 10 10 1
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation1
0
1
0 10 0 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation2
0
1
0 10 1 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation4
0
1
0 10 3 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation7
0
1
0 10 4 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 planet10
0
1
0 10 5 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star0
0
1
0 10 6 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star5
0
1
0 10 7 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star6
0
1
0 10 8 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star8
0
1
0 10 9 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star9
0
1
0 10 10 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation1
0
1
0 10 0 3
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation2
0
1
0 10 1 3
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation3
0
1
0 10 2 3
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation7
0
1
0 10 4 3
1
end_operator
begin_operator
turn_to satellite1 groundstation4 planet10
0
1
0 10 5 3
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star0
0
1
0 10 6 3
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star5
0
1
0 10 7 3
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star6
0
1
0 10 8 3
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star8
0
1
0 10 9 3
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star9
0
1
0 10 10 3
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation1
0
1
0 10 0 4
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation2
0
1
0 10 1 4
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation3
0
1
0 10 2 4
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation4
0
1
0 10 3 4
1
end_operator
begin_operator
turn_to satellite1 groundstation7 planet10
0
1
0 10 5 4
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star0
0
1
0 10 6 4
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star5
0
1
0 10 7 4
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star6
0
1
0 10 8 4
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star8
0
1
0 10 9 4
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star9
0
1
0 10 10 4
1
end_operator
begin_operator
turn_to satellite1 planet10 groundstation1
0
1
0 10 0 5
1
end_operator
begin_operator
turn_to satellite1 planet10 groundstation2
0
1
0 10 1 5
1
end_operator
begin_operator
turn_to satellite1 planet10 groundstation3
0
1
0 10 2 5
1
end_operator
begin_operator
turn_to satellite1 planet10 groundstation4
0
1
0 10 3 5
1
end_operator
begin_operator
turn_to satellite1 planet10 groundstation7
0
1
0 10 4 5
1
end_operator
begin_operator
turn_to satellite1 planet10 star0
0
1
0 10 6 5
1
end_operator
begin_operator
turn_to satellite1 planet10 star5
0
1
0 10 7 5
1
end_operator
begin_operator
turn_to satellite1 planet10 star6
0
1
0 10 8 5
1
end_operator
begin_operator
turn_to satellite1 planet10 star8
0
1
0 10 9 5
1
end_operator
begin_operator
turn_to satellite1 planet10 star9
0
1
0 10 10 5
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation1
0
1
0 10 0 6
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation2
0
1
0 10 1 6
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation3
0
1
0 10 2 6
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation4
0
1
0 10 3 6
1
end_operator
begin_operator
turn_to satellite1 star0 groundstation7
0
1
0 10 4 6
1
end_operator
begin_operator
turn_to satellite1 star0 planet10
0
1
0 10 5 6
1
end_operator
begin_operator
turn_to satellite1 star0 star5
0
1
0 10 7 6
1
end_operator
begin_operator
turn_to satellite1 star0 star6
0
1
0 10 8 6
1
end_operator
begin_operator
turn_to satellite1 star0 star8
0
1
0 10 9 6
1
end_operator
begin_operator
turn_to satellite1 star0 star9
0
1
0 10 10 6
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation1
0
1
0 10 0 7
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation2
0
1
0 10 1 7
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation3
0
1
0 10 2 7
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation4
0
1
0 10 3 7
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation7
0
1
0 10 4 7
1
end_operator
begin_operator
turn_to satellite1 star5 planet10
0
1
0 10 5 7
1
end_operator
begin_operator
turn_to satellite1 star5 star0
0
1
0 10 6 7
1
end_operator
begin_operator
turn_to satellite1 star5 star6
0
1
0 10 8 7
1
end_operator
begin_operator
turn_to satellite1 star5 star8
0
1
0 10 9 7
1
end_operator
begin_operator
turn_to satellite1 star5 star9
0
1
0 10 10 7
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation1
0
1
0 10 0 8
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation2
0
1
0 10 1 8
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation3
0
1
0 10 2 8
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation4
0
1
0 10 3 8
1
end_operator
begin_operator
turn_to satellite1 star6 groundstation7
0
1
0 10 4 8
1
end_operator
begin_operator
turn_to satellite1 star6 planet10
0
1
0 10 5 8
1
end_operator
begin_operator
turn_to satellite1 star6 star0
0
1
0 10 6 8
1
end_operator
begin_operator
turn_to satellite1 star6 star5
0
1
0 10 7 8
1
end_operator
begin_operator
turn_to satellite1 star6 star8
0
1
0 10 9 8
1
end_operator
begin_operator
turn_to satellite1 star6 star9
0
1
0 10 10 8
1
end_operator
begin_operator
turn_to satellite1 star8 groundstation1
0
1
0 10 0 9
1
end_operator
begin_operator
turn_to satellite1 star8 groundstation2
0
1
0 10 1 9
1
end_operator
begin_operator
turn_to satellite1 star8 groundstation3
0
1
0 10 2 9
1
end_operator
begin_operator
turn_to satellite1 star8 groundstation4
0
1
0 10 3 9
1
end_operator
begin_operator
turn_to satellite1 star8 groundstation7
0
1
0 10 4 9
1
end_operator
begin_operator
turn_to satellite1 star8 planet10
0
1
0 10 5 9
1
end_operator
begin_operator
turn_to satellite1 star8 star0
0
1
0 10 6 9
1
end_operator
begin_operator
turn_to satellite1 star8 star5
0
1
0 10 7 9
1
end_operator
begin_operator
turn_to satellite1 star8 star6
0
1
0 10 8 9
1
end_operator
begin_operator
turn_to satellite1 star8 star9
0
1
0 10 10 9
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation1
0
1
0 10 0 10
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation2
0
1
0 10 1 10
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation3
0
1
0 10 2 10
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation4
0
1
0 10 3 10
1
end_operator
begin_operator
turn_to satellite1 star9 groundstation7
0
1
0 10 4 10
1
end_operator
begin_operator
turn_to satellite1 star9 planet10
0
1
0 10 5 10
1
end_operator
begin_operator
turn_to satellite1 star9 star0
0
1
0 10 6 10
1
end_operator
begin_operator
turn_to satellite1 star9 star5
0
1
0 10 7 10
1
end_operator
begin_operator
turn_to satellite1 star9 star6
0
1
0 10 8 10
1
end_operator
begin_operator
turn_to satellite1 star9 star8
0
1
0 10 9 10
1
end_operator
begin_operator
turn_to satellite2 groundstation1 groundstation2
0
1
0 9 1 0
1
end_operator
begin_operator
turn_to satellite2 groundstation1 groundstation3
0
1
0 9 2 0
1
end_operator
begin_operator
turn_to satellite2 groundstation1 groundstation4
0
1
0 9 3 0
1
end_operator
begin_operator
turn_to satellite2 groundstation1 groundstation7
0
1
0 9 4 0
1
end_operator
begin_operator
turn_to satellite2 groundstation1 planet10
0
1
0 9 5 0
1
end_operator
begin_operator
turn_to satellite2 groundstation1 star0
0
1
0 9 6 0
1
end_operator
begin_operator
turn_to satellite2 groundstation1 star5
0
1
0 9 7 0
1
end_operator
begin_operator
turn_to satellite2 groundstation1 star6
0
1
0 9 8 0
1
end_operator
begin_operator
turn_to satellite2 groundstation1 star8
0
1
0 9 9 0
1
end_operator
begin_operator
turn_to satellite2 groundstation1 star9
0
1
0 9 10 0
1
end_operator
begin_operator
turn_to satellite2 groundstation2 groundstation1
0
1
0 9 0 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 groundstation3
0
1
0 9 2 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 groundstation4
0
1
0 9 3 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 groundstation7
0
1
0 9 4 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 planet10
0
1
0 9 5 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 star0
0
1
0 9 6 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 star5
0
1
0 9 7 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 star6
0
1
0 9 8 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 star8
0
1
0 9 9 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 star9
0
1
0 9 10 1
1
end_operator
begin_operator
turn_to satellite2 groundstation3 groundstation1
0
1
0 9 0 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 groundstation2
0
1
0 9 1 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 groundstation4
0
1
0 9 3 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 groundstation7
0
1
0 9 4 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 planet10
0
1
0 9 5 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star0
0
1
0 9 6 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star5
0
1
0 9 7 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star6
0
1
0 9 8 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star8
0
1
0 9 9 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star9
0
1
0 9 10 2
1
end_operator
begin_operator
turn_to satellite2 groundstation4 groundstation1
0
1
0 9 0 3
1
end_operator
begin_operator
turn_to satellite2 groundstation4 groundstation2
0
1
0 9 1 3
1
end_operator
begin_operator
turn_to satellite2 groundstation4 groundstation3
0
1
0 9 2 3
1
end_operator
begin_operator
turn_to satellite2 groundstation4 groundstation7
0
1
0 9 4 3
1
end_operator
begin_operator
turn_to satellite2 groundstation4 planet10
0
1
0 9 5 3
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star0
0
1
0 9 6 3
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star5
0
1
0 9 7 3
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star6
0
1
0 9 8 3
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star8
0
1
0 9 9 3
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star9
0
1
0 9 10 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 groundstation1
0
1
0 9 0 4
1
end_operator
begin_operator
turn_to satellite2 groundstation7 groundstation2
0
1
0 9 1 4
1
end_operator
begin_operator
turn_to satellite2 groundstation7 groundstation3
0
1
0 9 2 4
1
end_operator
begin_operator
turn_to satellite2 groundstation7 groundstation4
0
1
0 9 3 4
1
end_operator
begin_operator
turn_to satellite2 groundstation7 planet10
0
1
0 9 5 4
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star0
0
1
0 9 6 4
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star5
0
1
0 9 7 4
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star6
0
1
0 9 8 4
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star8
0
1
0 9 9 4
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star9
0
1
0 9 10 4
1
end_operator
begin_operator
turn_to satellite2 planet10 groundstation1
0
1
0 9 0 5
1
end_operator
begin_operator
turn_to satellite2 planet10 groundstation2
0
1
0 9 1 5
1
end_operator
begin_operator
turn_to satellite2 planet10 groundstation3
0
1
0 9 2 5
1
end_operator
begin_operator
turn_to satellite2 planet10 groundstation4
0
1
0 9 3 5
1
end_operator
begin_operator
turn_to satellite2 planet10 groundstation7
0
1
0 9 4 5
1
end_operator
begin_operator
turn_to satellite2 planet10 star0
0
1
0 9 6 5
1
end_operator
begin_operator
turn_to satellite2 planet10 star5
0
1
0 9 7 5
1
end_operator
begin_operator
turn_to satellite2 planet10 star6
0
1
0 9 8 5
1
end_operator
begin_operator
turn_to satellite2 planet10 star8
0
1
0 9 9 5
1
end_operator
begin_operator
turn_to satellite2 planet10 star9
0
1
0 9 10 5
1
end_operator
begin_operator
turn_to satellite2 star0 groundstation1
0
1
0 9 0 6
1
end_operator
begin_operator
turn_to satellite2 star0 groundstation2
0
1
0 9 1 6
1
end_operator
begin_operator
turn_to satellite2 star0 groundstation3
0
1
0 9 2 6
1
end_operator
begin_operator
turn_to satellite2 star0 groundstation4
0
1
0 9 3 6
1
end_operator
begin_operator
turn_to satellite2 star0 groundstation7
0
1
0 9 4 6
1
end_operator
begin_operator
turn_to satellite2 star0 planet10
0
1
0 9 5 6
1
end_operator
begin_operator
turn_to satellite2 star0 star5
0
1
0 9 7 6
1
end_operator
begin_operator
turn_to satellite2 star0 star6
0
1
0 9 8 6
1
end_operator
begin_operator
turn_to satellite2 star0 star8
0
1
0 9 9 6
1
end_operator
begin_operator
turn_to satellite2 star0 star9
0
1
0 9 10 6
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation1
0
1
0 9 0 7
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation2
0
1
0 9 1 7
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation3
0
1
0 9 2 7
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation4
0
1
0 9 3 7
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation7
0
1
0 9 4 7
1
end_operator
begin_operator
turn_to satellite2 star5 planet10
0
1
0 9 5 7
1
end_operator
begin_operator
turn_to satellite2 star5 star0
0
1
0 9 6 7
1
end_operator
begin_operator
turn_to satellite2 star5 star6
0
1
0 9 8 7
1
end_operator
begin_operator
turn_to satellite2 star5 star8
0
1
0 9 9 7
1
end_operator
begin_operator
turn_to satellite2 star5 star9
0
1
0 9 10 7
1
end_operator
begin_operator
turn_to satellite2 star6 groundstation1
0
1
0 9 0 8
1
end_operator
begin_operator
turn_to satellite2 star6 groundstation2
0
1
0 9 1 8
1
end_operator
begin_operator
turn_to satellite2 star6 groundstation3
0
1
0 9 2 8
1
end_operator
begin_operator
turn_to satellite2 star6 groundstation4
0
1
0 9 3 8
1
end_operator
begin_operator
turn_to satellite2 star6 groundstation7
0
1
0 9 4 8
1
end_operator
begin_operator
turn_to satellite2 star6 planet10
0
1
0 9 5 8
1
end_operator
begin_operator
turn_to satellite2 star6 star0
0
1
0 9 6 8
1
end_operator
begin_operator
turn_to satellite2 star6 star5
0
1
0 9 7 8
1
end_operator
begin_operator
turn_to satellite2 star6 star8
0
1
0 9 9 8
1
end_operator
begin_operator
turn_to satellite2 star6 star9
0
1
0 9 10 8
1
end_operator
begin_operator
turn_to satellite2 star8 groundstation1
0
1
0 9 0 9
1
end_operator
begin_operator
turn_to satellite2 star8 groundstation2
0
1
0 9 1 9
1
end_operator
begin_operator
turn_to satellite2 star8 groundstation3
0
1
0 9 2 9
1
end_operator
begin_operator
turn_to satellite2 star8 groundstation4
0
1
0 9 3 9
1
end_operator
begin_operator
turn_to satellite2 star8 groundstation7
0
1
0 9 4 9
1
end_operator
begin_operator
turn_to satellite2 star8 planet10
0
1
0 9 5 9
1
end_operator
begin_operator
turn_to satellite2 star8 star0
0
1
0 9 6 9
1
end_operator
begin_operator
turn_to satellite2 star8 star5
0
1
0 9 7 9
1
end_operator
begin_operator
turn_to satellite2 star8 star6
0
1
0 9 8 9
1
end_operator
begin_operator
turn_to satellite2 star8 star9
0
1
0 9 10 9
1
end_operator
begin_operator
turn_to satellite2 star9 groundstation1
0
1
0 9 0 10
1
end_operator
begin_operator
turn_to satellite2 star9 groundstation2
0
1
0 9 1 10
1
end_operator
begin_operator
turn_to satellite2 star9 groundstation3
0
1
0 9 2 10
1
end_operator
begin_operator
turn_to satellite2 star9 groundstation4
0
1
0 9 3 10
1
end_operator
begin_operator
turn_to satellite2 star9 groundstation7
0
1
0 9 4 10
1
end_operator
begin_operator
turn_to satellite2 star9 planet10
0
1
0 9 5 10
1
end_operator
begin_operator
turn_to satellite2 star9 star0
0
1
0 9 6 10
1
end_operator
begin_operator
turn_to satellite2 star9 star5
0
1
0 9 7 10
1
end_operator
begin_operator
turn_to satellite2 star9 star6
0
1
0 9 8 10
1
end_operator
begin_operator
turn_to satellite2 star9 star8
0
1
0 9 9 10
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation2
0
1
0 8 1 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation3
0
1
0 8 2 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation4
0
1
0 8 3 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation7
0
1
0 8 4 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 planet10
0
1
0 8 5 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 star0
0
1
0 8 6 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 star5
0
1
0 8 7 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 star6
0
1
0 8 8 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 star8
0
1
0 8 9 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 star9
0
1
0 8 10 0
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation1
0
1
0 8 0 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation3
0
1
0 8 2 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation4
0
1
0 8 3 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation7
0
1
0 8 4 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 planet10
0
1
0 8 5 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star0
0
1
0 8 6 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star5
0
1
0 8 7 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star6
0
1
0 8 8 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star8
0
1
0 8 9 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star9
0
1
0 8 10 1
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation1
0
1
0 8 0 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation2
0
1
0 8 1 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation4
0
1
0 8 3 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation7
0
1
0 8 4 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 planet10
0
1
0 8 5 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 star0
0
1
0 8 6 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 star5
0
1
0 8 7 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 star6
0
1
0 8 8 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 star8
0
1
0 8 9 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 star9
0
1
0 8 10 2
1
end_operator
begin_operator
turn_to satellite3 groundstation4 groundstation1
0
1
0 8 0 3
1
end_operator
begin_operator
turn_to satellite3 groundstation4 groundstation2
0
1
0 8 1 3
1
end_operator
begin_operator
turn_to satellite3 groundstation4 groundstation3
0
1
0 8 2 3
1
end_operator
begin_operator
turn_to satellite3 groundstation4 groundstation7
0
1
0 8 4 3
1
end_operator
begin_operator
turn_to satellite3 groundstation4 planet10
0
1
0 8 5 3
1
end_operator
begin_operator
turn_to satellite3 groundstation4 star0
0
1
0 8 6 3
1
end_operator
begin_operator
turn_to satellite3 groundstation4 star5
0
1
0 8 7 3
1
end_operator
begin_operator
turn_to satellite3 groundstation4 star6
0
1
0 8 8 3
1
end_operator
begin_operator
turn_to satellite3 groundstation4 star8
0
1
0 8 9 3
1
end_operator
begin_operator
turn_to satellite3 groundstation4 star9
0
1
0 8 10 3
1
end_operator
begin_operator
turn_to satellite3 groundstation7 groundstation1
0
1
0 8 0 4
1
end_operator
begin_operator
turn_to satellite3 groundstation7 groundstation2
0
1
0 8 1 4
1
end_operator
begin_operator
turn_to satellite3 groundstation7 groundstation3
0
1
0 8 2 4
1
end_operator
begin_operator
turn_to satellite3 groundstation7 groundstation4
0
1
0 8 3 4
1
end_operator
begin_operator
turn_to satellite3 groundstation7 planet10
0
1
0 8 5 4
1
end_operator
begin_operator
turn_to satellite3 groundstation7 star0
0
1
0 8 6 4
1
end_operator
begin_operator
turn_to satellite3 groundstation7 star5
0
1
0 8 7 4
1
end_operator
begin_operator
turn_to satellite3 groundstation7 star6
0
1
0 8 8 4
1
end_operator
begin_operator
turn_to satellite3 groundstation7 star8
0
1
0 8 9 4
1
end_operator
begin_operator
turn_to satellite3 groundstation7 star9
0
1
0 8 10 4
1
end_operator
begin_operator
turn_to satellite3 planet10 groundstation1
0
1
0 8 0 5
1
end_operator
begin_operator
turn_to satellite3 planet10 groundstation2
0
1
0 8 1 5
1
end_operator
begin_operator
turn_to satellite3 planet10 groundstation3
0
1
0 8 2 5
1
end_operator
begin_operator
turn_to satellite3 planet10 groundstation4
0
1
0 8 3 5
1
end_operator
begin_operator
turn_to satellite3 planet10 groundstation7
0
1
0 8 4 5
1
end_operator
begin_operator
turn_to satellite3 planet10 star0
0
1
0 8 6 5
1
end_operator
begin_operator
turn_to satellite3 planet10 star5
0
1
0 8 7 5
1
end_operator
begin_operator
turn_to satellite3 planet10 star6
0
1
0 8 8 5
1
end_operator
begin_operator
turn_to satellite3 planet10 star8
0
1
0 8 9 5
1
end_operator
begin_operator
turn_to satellite3 planet10 star9
0
1
0 8 10 5
1
end_operator
begin_operator
turn_to satellite3 star0 groundstation1
0
1
0 8 0 6
1
end_operator
begin_operator
turn_to satellite3 star0 groundstation2
0
1
0 8 1 6
1
end_operator
begin_operator
turn_to satellite3 star0 groundstation3
0
1
0 8 2 6
1
end_operator
begin_operator
turn_to satellite3 star0 groundstation4
0
1
0 8 3 6
1
end_operator
begin_operator
turn_to satellite3 star0 groundstation7
0
1
0 8 4 6
1
end_operator
begin_operator
turn_to satellite3 star0 planet10
0
1
0 8 5 6
1
end_operator
begin_operator
turn_to satellite3 star0 star5
0
1
0 8 7 6
1
end_operator
begin_operator
turn_to satellite3 star0 star6
0
1
0 8 8 6
1
end_operator
begin_operator
turn_to satellite3 star0 star8
0
1
0 8 9 6
1
end_operator
begin_operator
turn_to satellite3 star0 star9
0
1
0 8 10 6
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation1
0
1
0 8 0 7
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation2
0
1
0 8 1 7
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation3
0
1
0 8 2 7
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation4
0
1
0 8 3 7
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation7
0
1
0 8 4 7
1
end_operator
begin_operator
turn_to satellite3 star5 planet10
0
1
0 8 5 7
1
end_operator
begin_operator
turn_to satellite3 star5 star0
0
1
0 8 6 7
1
end_operator
begin_operator
turn_to satellite3 star5 star6
0
1
0 8 8 7
1
end_operator
begin_operator
turn_to satellite3 star5 star8
0
1
0 8 9 7
1
end_operator
begin_operator
turn_to satellite3 star5 star9
0
1
0 8 10 7
1
end_operator
begin_operator
turn_to satellite3 star6 groundstation1
0
1
0 8 0 8
1
end_operator
begin_operator
turn_to satellite3 star6 groundstation2
0
1
0 8 1 8
1
end_operator
begin_operator
turn_to satellite3 star6 groundstation3
0
1
0 8 2 8
1
end_operator
begin_operator
turn_to satellite3 star6 groundstation4
0
1
0 8 3 8
1
end_operator
begin_operator
turn_to satellite3 star6 groundstation7
0
1
0 8 4 8
1
end_operator
begin_operator
turn_to satellite3 star6 planet10
0
1
0 8 5 8
1
end_operator
begin_operator
turn_to satellite3 star6 star0
0
1
0 8 6 8
1
end_operator
begin_operator
turn_to satellite3 star6 star5
0
1
0 8 7 8
1
end_operator
begin_operator
turn_to satellite3 star6 star8
0
1
0 8 9 8
1
end_operator
begin_operator
turn_to satellite3 star6 star9
0
1
0 8 10 8
1
end_operator
begin_operator
turn_to satellite3 star8 groundstation1
0
1
0 8 0 9
1
end_operator
begin_operator
turn_to satellite3 star8 groundstation2
0
1
0 8 1 9
1
end_operator
begin_operator
turn_to satellite3 star8 groundstation3
0
1
0 8 2 9
1
end_operator
begin_operator
turn_to satellite3 star8 groundstation4
0
1
0 8 3 9
1
end_operator
begin_operator
turn_to satellite3 star8 groundstation7
0
1
0 8 4 9
1
end_operator
begin_operator
turn_to satellite3 star8 planet10
0
1
0 8 5 9
1
end_operator
begin_operator
turn_to satellite3 star8 star0
0
1
0 8 6 9
1
end_operator
begin_operator
turn_to satellite3 star8 star5
0
1
0 8 7 9
1
end_operator
begin_operator
turn_to satellite3 star8 star6
0
1
0 8 8 9
1
end_operator
begin_operator
turn_to satellite3 star8 star9
0
1
0 8 10 9
1
end_operator
begin_operator
turn_to satellite3 star9 groundstation1
0
1
0 8 0 10
1
end_operator
begin_operator
turn_to satellite3 star9 groundstation2
0
1
0 8 1 10
1
end_operator
begin_operator
turn_to satellite3 star9 groundstation3
0
1
0 8 2 10
1
end_operator
begin_operator
turn_to satellite3 star9 groundstation4
0
1
0 8 3 10
1
end_operator
begin_operator
turn_to satellite3 star9 groundstation7
0
1
0 8 4 10
1
end_operator
begin_operator
turn_to satellite3 star9 planet10
0
1
0 8 5 10
1
end_operator
begin_operator
turn_to satellite3 star9 star0
0
1
0 8 6 10
1
end_operator
begin_operator
turn_to satellite3 star9 star5
0
1
0 8 7 10
1
end_operator
begin_operator
turn_to satellite3 star9 star6
0
1
0 8 8 10
1
end_operator
begin_operator
turn_to satellite3 star9 star8
0
1
0 8 9 10
1
end_operator
0
