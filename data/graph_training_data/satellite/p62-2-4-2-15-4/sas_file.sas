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
Atom power_on(instrument1)
NegatedAtom power_on(instrument1)
end_variable
begin_variable
var1
-1
2
Atom power_on(instrument2)
NegatedAtom power_on(instrument2)
end_variable
begin_variable
var2
-1
2
Atom power_on(instrument3)
NegatedAtom power_on(instrument3)
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
Atom power_avail(satellite0)
NegatedAtom power_avail(satellite0)
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
19
Atom pointing(satellite1, groundstation0)
Atom pointing(satellite1, groundstation12)
Atom pointing(satellite1, groundstation14)
Atom pointing(satellite1, groundstation2)
Atom pointing(satellite1, groundstation6)
Atom pointing(satellite1, groundstation7)
Atom pointing(satellite1, groundstation9)
Atom pointing(satellite1, phenomenon16)
Atom pointing(satellite1, planet15)
Atom pointing(satellite1, star1)
Atom pointing(satellite1, star10)
Atom pointing(satellite1, star11)
Atom pointing(satellite1, star13)
Atom pointing(satellite1, star17)
Atom pointing(satellite1, star18)
Atom pointing(satellite1, star3)
Atom pointing(satellite1, star4)
Atom pointing(satellite1, star5)
Atom pointing(satellite1, star8)
end_variable
begin_variable
var7
-1
19
Atom pointing(satellite0, groundstation0)
Atom pointing(satellite0, groundstation12)
Atom pointing(satellite0, groundstation14)
Atom pointing(satellite0, groundstation2)
Atom pointing(satellite0, groundstation6)
Atom pointing(satellite0, groundstation7)
Atom pointing(satellite0, groundstation9)
Atom pointing(satellite0, phenomenon16)
Atom pointing(satellite0, planet15)
Atom pointing(satellite0, star1)
Atom pointing(satellite0, star10)
Atom pointing(satellite0, star11)
Atom pointing(satellite0, star13)
Atom pointing(satellite0, star17)
Atom pointing(satellite0, star18)
Atom pointing(satellite0, star3)
Atom pointing(satellite0, star4)
Atom pointing(satellite0, star5)
Atom pointing(satellite0, star8)
end_variable
begin_variable
var8
-1
2
Atom calibrated(instrument3)
NegatedAtom calibrated(instrument3)
end_variable
begin_variable
var9
-1
2
Atom calibrated(instrument2)
NegatedAtom calibrated(instrument2)
end_variable
begin_variable
var10
-1
2
Atom calibrated(instrument1)
NegatedAtom calibrated(instrument1)
end_variable
begin_variable
var11
-1
2
Atom calibrated(instrument0)
NegatedAtom calibrated(instrument0)
end_variable
begin_variable
var12
-1
2
Atom have_image(star18, spectrograph0)
NegatedAtom have_image(star18, spectrograph0)
end_variable
begin_variable
var13
-1
2
Atom have_image(planet15, spectrograph0)
NegatedAtom have_image(planet15, spectrograph0)
end_variable
0
begin_state
1
1
1
0
0
1
16
4
1
1
1
1
1
1
end_state
begin_goal
2
12 0
13 0
end_goal
711
begin_operator
calibrate satellite0 instrument0 groundstation7
2
7 5
5 0
1
0 11 -1 0
1
end_operator
begin_operator
calibrate satellite0 instrument0 star1
2
7 9
5 0
1
0 11 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument1 groundstation7
2
6 5
0 0
1
0 10 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument1 groundstation9
2
6 6
0 0
1
0 10 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument1 star3
2
6 15
0 0
1
0 10 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument2 groundstation2
2
6 3
1 0
1
0 9 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument2 groundstation9
2
6 6
1 0
1
0 9 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument2 star1
2
6 9
1 0
1
0 9 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument3 groundstation14
2
6 2
2 0
1
0 8 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument3 star11
2
6 11
2 0
1
0 8 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument3 star3
2
6 15
2 0
1
0 8 -1 0
1
end_operator
begin_operator
switch_off instrument0 satellite0
0
2
0 4 -1 0
0 5 0 1
1
end_operator
begin_operator
switch_off instrument1 satellite1
0
2
0 3 -1 0
0 0 0 1
1
end_operator
begin_operator
switch_off instrument2 satellite1
0
2
0 3 -1 0
0 1 0 1
1
end_operator
begin_operator
switch_off instrument3 satellite1
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
0 11 -1 1
0 4 0 1
0 5 -1 0
1
end_operator
begin_operator
switch_on instrument1 satellite1
0
3
0 10 -1 1
0 3 0 1
0 0 -1 0
1
end_operator
begin_operator
switch_on instrument2 satellite1
0
3
0 9 -1 1
0 3 0 1
0 1 -1 0
1
end_operator
begin_operator
switch_on instrument3 satellite1
0
3
0 8 -1 1
0 3 0 1
0 2 -1 0
1
end_operator
begin_operator
take_image satellite0 planet15 instrument0 spectrograph0
3
11 0
7 8
5 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite0 star18 instrument0 spectrograph0
3
11 0
7 14
5 0
1
0 12 -1 0
1
end_operator
begin_operator
take_image satellite1 planet15 instrument1 spectrograph0
3
10 0
6 8
0 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite1 planet15 instrument2 spectrograph0
3
9 0
6 8
1 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite1 planet15 instrument3 spectrograph0
3
8 0
6 8
2 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite1 star18 instrument1 spectrograph0
3
10 0
6 14
0 0
1
0 12 -1 0
1
end_operator
begin_operator
take_image satellite1 star18 instrument2 spectrograph0
3
9 0
6 14
1 0
1
0 12 -1 0
1
end_operator
begin_operator
take_image satellite1 star18 instrument3 spectrograph0
3
8 0
6 14
2 0
1
0 12 -1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation12
0
1
0 7 1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation14
0
1
0 7 2 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation2
0
1
0 7 3 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation6
0
1
0 7 4 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation7
0
1
0 7 5 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation9
0
1
0 7 6 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 phenomenon16
0
1
0 7 7 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 planet15
0
1
0 7 8 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star1
0
1
0 7 9 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star10
0
1
0 7 10 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star11
0
1
0 7 11 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star13
0
1
0 7 12 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star17
0
1
0 7 13 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star18
0
1
0 7 14 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star3
0
1
0 7 15 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star4
0
1
0 7 16 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star5
0
1
0 7 17 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star8
0
1
0 7 18 0
1
end_operator
begin_operator
turn_to satellite0 groundstation12 groundstation0
0
1
0 7 0 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 groundstation14
0
1
0 7 2 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 groundstation2
0
1
0 7 3 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 groundstation6
0
1
0 7 4 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 groundstation7
0
1
0 7 5 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 groundstation9
0
1
0 7 6 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 phenomenon16
0
1
0 7 7 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 planet15
0
1
0 7 8 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 star1
0
1
0 7 9 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 star10
0
1
0 7 10 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 star11
0
1
0 7 11 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 star13
0
1
0 7 12 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 star17
0
1
0 7 13 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 star18
0
1
0 7 14 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 star3
0
1
0 7 15 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 star4
0
1
0 7 16 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 star5
0
1
0 7 17 1
1
end_operator
begin_operator
turn_to satellite0 groundstation12 star8
0
1
0 7 18 1
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation0
0
1
0 7 0 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation12
0
1
0 7 1 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation2
0
1
0 7 3 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation6
0
1
0 7 4 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation7
0
1
0 7 5 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 groundstation9
0
1
0 7 6 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 phenomenon16
0
1
0 7 7 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 planet15
0
1
0 7 8 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star1
0
1
0 7 9 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star10
0
1
0 7 10 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star11
0
1
0 7 11 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star13
0
1
0 7 12 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star17
0
1
0 7 13 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star18
0
1
0 7 14 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star3
0
1
0 7 15 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star4
0
1
0 7 16 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star5
0
1
0 7 17 2
1
end_operator
begin_operator
turn_to satellite0 groundstation14 star8
0
1
0 7 18 2
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation0
0
1
0 7 0 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation12
0
1
0 7 1 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation14
0
1
0 7 2 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation6
0
1
0 7 4 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation7
0
1
0 7 5 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation9
0
1
0 7 6 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 phenomenon16
0
1
0 7 7 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 planet15
0
1
0 7 8 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star1
0
1
0 7 9 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star10
0
1
0 7 10 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star11
0
1
0 7 11 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star13
0
1
0 7 12 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star17
0
1
0 7 13 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star18
0
1
0 7 14 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star3
0
1
0 7 15 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star4
0
1
0 7 16 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star5
0
1
0 7 17 3
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star8
0
1
0 7 18 3
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation0
0
1
0 7 0 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation12
0
1
0 7 1 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation14
0
1
0 7 2 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation2
0
1
0 7 3 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation7
0
1
0 7 5 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 groundstation9
0
1
0 7 6 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 phenomenon16
0
1
0 7 7 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 planet15
0
1
0 7 8 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star1
0
1
0 7 9 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star10
0
1
0 7 10 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star11
0
1
0 7 11 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star13
0
1
0 7 12 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star17
0
1
0 7 13 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star18
0
1
0 7 14 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star3
0
1
0 7 15 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star4
0
1
0 7 16 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star5
0
1
0 7 17 4
1
end_operator
begin_operator
turn_to satellite0 groundstation6 star8
0
1
0 7 18 4
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation0
0
1
0 7 0 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation12
0
1
0 7 1 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation14
0
1
0 7 2 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation2
0
1
0 7 3 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation6
0
1
0 7 4 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 groundstation9
0
1
0 7 6 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 phenomenon16
0
1
0 7 7 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 planet15
0
1
0 7 8 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star1
0
1
0 7 9 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star10
0
1
0 7 10 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star11
0
1
0 7 11 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star13
0
1
0 7 12 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star17
0
1
0 7 13 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star18
0
1
0 7 14 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star3
0
1
0 7 15 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star4
0
1
0 7 16 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star5
0
1
0 7 17 5
1
end_operator
begin_operator
turn_to satellite0 groundstation7 star8
0
1
0 7 18 5
1
end_operator
begin_operator
turn_to satellite0 groundstation9 groundstation0
0
1
0 7 0 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 groundstation12
0
1
0 7 1 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 groundstation14
0
1
0 7 2 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 groundstation2
0
1
0 7 3 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 groundstation6
0
1
0 7 4 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 groundstation7
0
1
0 7 5 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 phenomenon16
0
1
0 7 7 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 planet15
0
1
0 7 8 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star1
0
1
0 7 9 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star10
0
1
0 7 10 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star11
0
1
0 7 11 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star13
0
1
0 7 12 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star17
0
1
0 7 13 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star18
0
1
0 7 14 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star3
0
1
0 7 15 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star4
0
1
0 7 16 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star5
0
1
0 7 17 6
1
end_operator
begin_operator
turn_to satellite0 groundstation9 star8
0
1
0 7 18 6
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 groundstation0
0
1
0 7 0 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 groundstation12
0
1
0 7 1 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 groundstation14
0
1
0 7 2 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 groundstation2
0
1
0 7 3 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 groundstation6
0
1
0 7 4 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 groundstation7
0
1
0 7 5 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 groundstation9
0
1
0 7 6 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 planet15
0
1
0 7 8 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 star1
0
1
0 7 9 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 star10
0
1
0 7 10 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 star11
0
1
0 7 11 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 star13
0
1
0 7 12 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 star17
0
1
0 7 13 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 star18
0
1
0 7 14 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 star3
0
1
0 7 15 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 star4
0
1
0 7 16 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 star5
0
1
0 7 17 7
1
end_operator
begin_operator
turn_to satellite0 phenomenon16 star8
0
1
0 7 18 7
1
end_operator
begin_operator
turn_to satellite0 planet15 groundstation0
0
1
0 7 0 8
1
end_operator
begin_operator
turn_to satellite0 planet15 groundstation12
0
1
0 7 1 8
1
end_operator
begin_operator
turn_to satellite0 planet15 groundstation14
0
1
0 7 2 8
1
end_operator
begin_operator
turn_to satellite0 planet15 groundstation2
0
1
0 7 3 8
1
end_operator
begin_operator
turn_to satellite0 planet15 groundstation6
0
1
0 7 4 8
1
end_operator
begin_operator
turn_to satellite0 planet15 groundstation7
0
1
0 7 5 8
1
end_operator
begin_operator
turn_to satellite0 planet15 groundstation9
0
1
0 7 6 8
1
end_operator
begin_operator
turn_to satellite0 planet15 phenomenon16
0
1
0 7 7 8
1
end_operator
begin_operator
turn_to satellite0 planet15 star1
0
1
0 7 9 8
1
end_operator
begin_operator
turn_to satellite0 planet15 star10
0
1
0 7 10 8
1
end_operator
begin_operator
turn_to satellite0 planet15 star11
0
1
0 7 11 8
1
end_operator
begin_operator
turn_to satellite0 planet15 star13
0
1
0 7 12 8
1
end_operator
begin_operator
turn_to satellite0 planet15 star17
0
1
0 7 13 8
1
end_operator
begin_operator
turn_to satellite0 planet15 star18
0
1
0 7 14 8
1
end_operator
begin_operator
turn_to satellite0 planet15 star3
0
1
0 7 15 8
1
end_operator
begin_operator
turn_to satellite0 planet15 star4
0
1
0 7 16 8
1
end_operator
begin_operator
turn_to satellite0 planet15 star5
0
1
0 7 17 8
1
end_operator
begin_operator
turn_to satellite0 planet15 star8
0
1
0 7 18 8
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation0
0
1
0 7 0 9
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation12
0
1
0 7 1 9
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation14
0
1
0 7 2 9
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation2
0
1
0 7 3 9
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation6
0
1
0 7 4 9
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation7
0
1
0 7 5 9
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation9
0
1
0 7 6 9
1
end_operator
begin_operator
turn_to satellite0 star1 phenomenon16
0
1
0 7 7 9
1
end_operator
begin_operator
turn_to satellite0 star1 planet15
0
1
0 7 8 9
1
end_operator
begin_operator
turn_to satellite0 star1 star10
0
1
0 7 10 9
1
end_operator
begin_operator
turn_to satellite0 star1 star11
0
1
0 7 11 9
1
end_operator
begin_operator
turn_to satellite0 star1 star13
0
1
0 7 12 9
1
end_operator
begin_operator
turn_to satellite0 star1 star17
0
1
0 7 13 9
1
end_operator
begin_operator
turn_to satellite0 star1 star18
0
1
0 7 14 9
1
end_operator
begin_operator
turn_to satellite0 star1 star3
0
1
0 7 15 9
1
end_operator
begin_operator
turn_to satellite0 star1 star4
0
1
0 7 16 9
1
end_operator
begin_operator
turn_to satellite0 star1 star5
0
1
0 7 17 9
1
end_operator
begin_operator
turn_to satellite0 star1 star8
0
1
0 7 18 9
1
end_operator
begin_operator
turn_to satellite0 star10 groundstation0
0
1
0 7 0 10
1
end_operator
begin_operator
turn_to satellite0 star10 groundstation12
0
1
0 7 1 10
1
end_operator
begin_operator
turn_to satellite0 star10 groundstation14
0
1
0 7 2 10
1
end_operator
begin_operator
turn_to satellite0 star10 groundstation2
0
1
0 7 3 10
1
end_operator
begin_operator
turn_to satellite0 star10 groundstation6
0
1
0 7 4 10
1
end_operator
begin_operator
turn_to satellite0 star10 groundstation7
0
1
0 7 5 10
1
end_operator
begin_operator
turn_to satellite0 star10 groundstation9
0
1
0 7 6 10
1
end_operator
begin_operator
turn_to satellite0 star10 phenomenon16
0
1
0 7 7 10
1
end_operator
begin_operator
turn_to satellite0 star10 planet15
0
1
0 7 8 10
1
end_operator
begin_operator
turn_to satellite0 star10 star1
0
1
0 7 9 10
1
end_operator
begin_operator
turn_to satellite0 star10 star11
0
1
0 7 11 10
1
end_operator
begin_operator
turn_to satellite0 star10 star13
0
1
0 7 12 10
1
end_operator
begin_operator
turn_to satellite0 star10 star17
0
1
0 7 13 10
1
end_operator
begin_operator
turn_to satellite0 star10 star18
0
1
0 7 14 10
1
end_operator
begin_operator
turn_to satellite0 star10 star3
0
1
0 7 15 10
1
end_operator
begin_operator
turn_to satellite0 star10 star4
0
1
0 7 16 10
1
end_operator
begin_operator
turn_to satellite0 star10 star5
0
1
0 7 17 10
1
end_operator
begin_operator
turn_to satellite0 star10 star8
0
1
0 7 18 10
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation0
0
1
0 7 0 11
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation12
0
1
0 7 1 11
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation14
0
1
0 7 2 11
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation2
0
1
0 7 3 11
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation6
0
1
0 7 4 11
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation7
0
1
0 7 5 11
1
end_operator
begin_operator
turn_to satellite0 star11 groundstation9
0
1
0 7 6 11
1
end_operator
begin_operator
turn_to satellite0 star11 phenomenon16
0
1
0 7 7 11
1
end_operator
begin_operator
turn_to satellite0 star11 planet15
0
1
0 7 8 11
1
end_operator
begin_operator
turn_to satellite0 star11 star1
0
1
0 7 9 11
1
end_operator
begin_operator
turn_to satellite0 star11 star10
0
1
0 7 10 11
1
end_operator
begin_operator
turn_to satellite0 star11 star13
0
1
0 7 12 11
1
end_operator
begin_operator
turn_to satellite0 star11 star17
0
1
0 7 13 11
1
end_operator
begin_operator
turn_to satellite0 star11 star18
0
1
0 7 14 11
1
end_operator
begin_operator
turn_to satellite0 star11 star3
0
1
0 7 15 11
1
end_operator
begin_operator
turn_to satellite0 star11 star4
0
1
0 7 16 11
1
end_operator
begin_operator
turn_to satellite0 star11 star5
0
1
0 7 17 11
1
end_operator
begin_operator
turn_to satellite0 star11 star8
0
1
0 7 18 11
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation0
0
1
0 7 0 12
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation12
0
1
0 7 1 12
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation14
0
1
0 7 2 12
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation2
0
1
0 7 3 12
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation6
0
1
0 7 4 12
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation7
0
1
0 7 5 12
1
end_operator
begin_operator
turn_to satellite0 star13 groundstation9
0
1
0 7 6 12
1
end_operator
begin_operator
turn_to satellite0 star13 phenomenon16
0
1
0 7 7 12
1
end_operator
begin_operator
turn_to satellite0 star13 planet15
0
1
0 7 8 12
1
end_operator
begin_operator
turn_to satellite0 star13 star1
0
1
0 7 9 12
1
end_operator
begin_operator
turn_to satellite0 star13 star10
0
1
0 7 10 12
1
end_operator
begin_operator
turn_to satellite0 star13 star11
0
1
0 7 11 12
1
end_operator
begin_operator
turn_to satellite0 star13 star17
0
1
0 7 13 12
1
end_operator
begin_operator
turn_to satellite0 star13 star18
0
1
0 7 14 12
1
end_operator
begin_operator
turn_to satellite0 star13 star3
0
1
0 7 15 12
1
end_operator
begin_operator
turn_to satellite0 star13 star4
0
1
0 7 16 12
1
end_operator
begin_operator
turn_to satellite0 star13 star5
0
1
0 7 17 12
1
end_operator
begin_operator
turn_to satellite0 star13 star8
0
1
0 7 18 12
1
end_operator
begin_operator
turn_to satellite0 star17 groundstation0
0
1
0 7 0 13
1
end_operator
begin_operator
turn_to satellite0 star17 groundstation12
0
1
0 7 1 13
1
end_operator
begin_operator
turn_to satellite0 star17 groundstation14
0
1
0 7 2 13
1
end_operator
begin_operator
turn_to satellite0 star17 groundstation2
0
1
0 7 3 13
1
end_operator
begin_operator
turn_to satellite0 star17 groundstation6
0
1
0 7 4 13
1
end_operator
begin_operator
turn_to satellite0 star17 groundstation7
0
1
0 7 5 13
1
end_operator
begin_operator
turn_to satellite0 star17 groundstation9
0
1
0 7 6 13
1
end_operator
begin_operator
turn_to satellite0 star17 phenomenon16
0
1
0 7 7 13
1
end_operator
begin_operator
turn_to satellite0 star17 planet15
0
1
0 7 8 13
1
end_operator
begin_operator
turn_to satellite0 star17 star1
0
1
0 7 9 13
1
end_operator
begin_operator
turn_to satellite0 star17 star10
0
1
0 7 10 13
1
end_operator
begin_operator
turn_to satellite0 star17 star11
0
1
0 7 11 13
1
end_operator
begin_operator
turn_to satellite0 star17 star13
0
1
0 7 12 13
1
end_operator
begin_operator
turn_to satellite0 star17 star18
0
1
0 7 14 13
1
end_operator
begin_operator
turn_to satellite0 star17 star3
0
1
0 7 15 13
1
end_operator
begin_operator
turn_to satellite0 star17 star4
0
1
0 7 16 13
1
end_operator
begin_operator
turn_to satellite0 star17 star5
0
1
0 7 17 13
1
end_operator
begin_operator
turn_to satellite0 star17 star8
0
1
0 7 18 13
1
end_operator
begin_operator
turn_to satellite0 star18 groundstation0
0
1
0 7 0 14
1
end_operator
begin_operator
turn_to satellite0 star18 groundstation12
0
1
0 7 1 14
1
end_operator
begin_operator
turn_to satellite0 star18 groundstation14
0
1
0 7 2 14
1
end_operator
begin_operator
turn_to satellite0 star18 groundstation2
0
1
0 7 3 14
1
end_operator
begin_operator
turn_to satellite0 star18 groundstation6
0
1
0 7 4 14
1
end_operator
begin_operator
turn_to satellite0 star18 groundstation7
0
1
0 7 5 14
1
end_operator
begin_operator
turn_to satellite0 star18 groundstation9
0
1
0 7 6 14
1
end_operator
begin_operator
turn_to satellite0 star18 phenomenon16
0
1
0 7 7 14
1
end_operator
begin_operator
turn_to satellite0 star18 planet15
0
1
0 7 8 14
1
end_operator
begin_operator
turn_to satellite0 star18 star1
0
1
0 7 9 14
1
end_operator
begin_operator
turn_to satellite0 star18 star10
0
1
0 7 10 14
1
end_operator
begin_operator
turn_to satellite0 star18 star11
0
1
0 7 11 14
1
end_operator
begin_operator
turn_to satellite0 star18 star13
0
1
0 7 12 14
1
end_operator
begin_operator
turn_to satellite0 star18 star17
0
1
0 7 13 14
1
end_operator
begin_operator
turn_to satellite0 star18 star3
0
1
0 7 15 14
1
end_operator
begin_operator
turn_to satellite0 star18 star4
0
1
0 7 16 14
1
end_operator
begin_operator
turn_to satellite0 star18 star5
0
1
0 7 17 14
1
end_operator
begin_operator
turn_to satellite0 star18 star8
0
1
0 7 18 14
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation0
0
1
0 7 0 15
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation12
0
1
0 7 1 15
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation14
0
1
0 7 2 15
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation2
0
1
0 7 3 15
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation6
0
1
0 7 4 15
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation7
0
1
0 7 5 15
1
end_operator
begin_operator
turn_to satellite0 star3 groundstation9
0
1
0 7 6 15
1
end_operator
begin_operator
turn_to satellite0 star3 phenomenon16
0
1
0 7 7 15
1
end_operator
begin_operator
turn_to satellite0 star3 planet15
0
1
0 7 8 15
1
end_operator
begin_operator
turn_to satellite0 star3 star1
0
1
0 7 9 15
1
end_operator
begin_operator
turn_to satellite0 star3 star10
0
1
0 7 10 15
1
end_operator
begin_operator
turn_to satellite0 star3 star11
0
1
0 7 11 15
1
end_operator
begin_operator
turn_to satellite0 star3 star13
0
1
0 7 12 15
1
end_operator
begin_operator
turn_to satellite0 star3 star17
0
1
0 7 13 15
1
end_operator
begin_operator
turn_to satellite0 star3 star18
0
1
0 7 14 15
1
end_operator
begin_operator
turn_to satellite0 star3 star4
0
1
0 7 16 15
1
end_operator
begin_operator
turn_to satellite0 star3 star5
0
1
0 7 17 15
1
end_operator
begin_operator
turn_to satellite0 star3 star8
0
1
0 7 18 15
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation0
0
1
0 7 0 16
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation12
0
1
0 7 1 16
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation14
0
1
0 7 2 16
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation2
0
1
0 7 3 16
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation6
0
1
0 7 4 16
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation7
0
1
0 7 5 16
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation9
0
1
0 7 6 16
1
end_operator
begin_operator
turn_to satellite0 star4 phenomenon16
0
1
0 7 7 16
1
end_operator
begin_operator
turn_to satellite0 star4 planet15
0
1
0 7 8 16
1
end_operator
begin_operator
turn_to satellite0 star4 star1
0
1
0 7 9 16
1
end_operator
begin_operator
turn_to satellite0 star4 star10
0
1
0 7 10 16
1
end_operator
begin_operator
turn_to satellite0 star4 star11
0
1
0 7 11 16
1
end_operator
begin_operator
turn_to satellite0 star4 star13
0
1
0 7 12 16
1
end_operator
begin_operator
turn_to satellite0 star4 star17
0
1
0 7 13 16
1
end_operator
begin_operator
turn_to satellite0 star4 star18
0
1
0 7 14 16
1
end_operator
begin_operator
turn_to satellite0 star4 star3
0
1
0 7 15 16
1
end_operator
begin_operator
turn_to satellite0 star4 star5
0
1
0 7 17 16
1
end_operator
begin_operator
turn_to satellite0 star4 star8
0
1
0 7 18 16
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation0
0
1
0 7 0 17
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation12
0
1
0 7 1 17
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation14
0
1
0 7 2 17
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation2
0
1
0 7 3 17
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation6
0
1
0 7 4 17
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation7
0
1
0 7 5 17
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation9
0
1
0 7 6 17
1
end_operator
begin_operator
turn_to satellite0 star5 phenomenon16
0
1
0 7 7 17
1
end_operator
begin_operator
turn_to satellite0 star5 planet15
0
1
0 7 8 17
1
end_operator
begin_operator
turn_to satellite0 star5 star1
0
1
0 7 9 17
1
end_operator
begin_operator
turn_to satellite0 star5 star10
0
1
0 7 10 17
1
end_operator
begin_operator
turn_to satellite0 star5 star11
0
1
0 7 11 17
1
end_operator
begin_operator
turn_to satellite0 star5 star13
0
1
0 7 12 17
1
end_operator
begin_operator
turn_to satellite0 star5 star17
0
1
0 7 13 17
1
end_operator
begin_operator
turn_to satellite0 star5 star18
0
1
0 7 14 17
1
end_operator
begin_operator
turn_to satellite0 star5 star3
0
1
0 7 15 17
1
end_operator
begin_operator
turn_to satellite0 star5 star4
0
1
0 7 16 17
1
end_operator
begin_operator
turn_to satellite0 star5 star8
0
1
0 7 18 17
1
end_operator
begin_operator
turn_to satellite0 star8 groundstation0
0
1
0 7 0 18
1
end_operator
begin_operator
turn_to satellite0 star8 groundstation12
0
1
0 7 1 18
1
end_operator
begin_operator
turn_to satellite0 star8 groundstation14
0
1
0 7 2 18
1
end_operator
begin_operator
turn_to satellite0 star8 groundstation2
0
1
0 7 3 18
1
end_operator
begin_operator
turn_to satellite0 star8 groundstation6
0
1
0 7 4 18
1
end_operator
begin_operator
turn_to satellite0 star8 groundstation7
0
1
0 7 5 18
1
end_operator
begin_operator
turn_to satellite0 star8 groundstation9
0
1
0 7 6 18
1
end_operator
begin_operator
turn_to satellite0 star8 phenomenon16
0
1
0 7 7 18
1
end_operator
begin_operator
turn_to satellite0 star8 planet15
0
1
0 7 8 18
1
end_operator
begin_operator
turn_to satellite0 star8 star1
0
1
0 7 9 18
1
end_operator
begin_operator
turn_to satellite0 star8 star10
0
1
0 7 10 18
1
end_operator
begin_operator
turn_to satellite0 star8 star11
0
1
0 7 11 18
1
end_operator
begin_operator
turn_to satellite0 star8 star13
0
1
0 7 12 18
1
end_operator
begin_operator
turn_to satellite0 star8 star17
0
1
0 7 13 18
1
end_operator
begin_operator
turn_to satellite0 star8 star18
0
1
0 7 14 18
1
end_operator
begin_operator
turn_to satellite0 star8 star3
0
1
0 7 15 18
1
end_operator
begin_operator
turn_to satellite0 star8 star4
0
1
0 7 16 18
1
end_operator
begin_operator
turn_to satellite0 star8 star5
0
1
0 7 17 18
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation12
0
1
0 6 1 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation14
0
1
0 6 2 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation2
0
1
0 6 3 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation6
0
1
0 6 4 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation7
0
1
0 6 5 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation9
0
1
0 6 6 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 phenomenon16
0
1
0 6 7 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 planet15
0
1
0 6 8 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star1
0
1
0 6 9 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star10
0
1
0 6 10 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star11
0
1
0 6 11 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star13
0
1
0 6 12 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star17
0
1
0 6 13 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star18
0
1
0 6 14 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star3
0
1
0 6 15 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star4
0
1
0 6 16 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star5
0
1
0 6 17 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star8
0
1
0 6 18 0
1
end_operator
begin_operator
turn_to satellite1 groundstation12 groundstation0
0
1
0 6 0 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 groundstation14
0
1
0 6 2 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 groundstation2
0
1
0 6 3 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 groundstation6
0
1
0 6 4 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 groundstation7
0
1
0 6 5 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 groundstation9
0
1
0 6 6 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 phenomenon16
0
1
0 6 7 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 planet15
0
1
0 6 8 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 star1
0
1
0 6 9 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 star10
0
1
0 6 10 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 star11
0
1
0 6 11 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 star13
0
1
0 6 12 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 star17
0
1
0 6 13 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 star18
0
1
0 6 14 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 star3
0
1
0 6 15 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 star4
0
1
0 6 16 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 star5
0
1
0 6 17 1
1
end_operator
begin_operator
turn_to satellite1 groundstation12 star8
0
1
0 6 18 1
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation0
0
1
0 6 0 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation12
0
1
0 6 1 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation2
0
1
0 6 3 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation6
0
1
0 6 4 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation7
0
1
0 6 5 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 groundstation9
0
1
0 6 6 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 phenomenon16
0
1
0 6 7 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 planet15
0
1
0 6 8 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star1
0
1
0 6 9 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star10
0
1
0 6 10 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star11
0
1
0 6 11 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star13
0
1
0 6 12 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star17
0
1
0 6 13 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star18
0
1
0 6 14 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star3
0
1
0 6 15 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star4
0
1
0 6 16 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star5
0
1
0 6 17 2
1
end_operator
begin_operator
turn_to satellite1 groundstation14 star8
0
1
0 6 18 2
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation0
0
1
0 6 0 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation12
0
1
0 6 1 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation14
0
1
0 6 2 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation6
0
1
0 6 4 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation7
0
1
0 6 5 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation9
0
1
0 6 6 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 phenomenon16
0
1
0 6 7 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 planet15
0
1
0 6 8 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star1
0
1
0 6 9 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star10
0
1
0 6 10 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star11
0
1
0 6 11 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star13
0
1
0 6 12 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star17
0
1
0 6 13 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star18
0
1
0 6 14 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star3
0
1
0 6 15 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star4
0
1
0 6 16 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star5
0
1
0 6 17 3
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star8
0
1
0 6 18 3
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation0
0
1
0 6 0 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation12
0
1
0 6 1 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation14
0
1
0 6 2 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation2
0
1
0 6 3 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation7
0
1
0 6 5 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 groundstation9
0
1
0 6 6 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 phenomenon16
0
1
0 6 7 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 planet15
0
1
0 6 8 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star1
0
1
0 6 9 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star10
0
1
0 6 10 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star11
0
1
0 6 11 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star13
0
1
0 6 12 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star17
0
1
0 6 13 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star18
0
1
0 6 14 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star3
0
1
0 6 15 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star4
0
1
0 6 16 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star5
0
1
0 6 17 4
1
end_operator
begin_operator
turn_to satellite1 groundstation6 star8
0
1
0 6 18 4
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation0
0
1
0 6 0 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation12
0
1
0 6 1 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation14
0
1
0 6 2 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation2
0
1
0 6 3 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation6
0
1
0 6 4 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 groundstation9
0
1
0 6 6 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 phenomenon16
0
1
0 6 7 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 planet15
0
1
0 6 8 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star1
0
1
0 6 9 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star10
0
1
0 6 10 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star11
0
1
0 6 11 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star13
0
1
0 6 12 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star17
0
1
0 6 13 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star18
0
1
0 6 14 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star3
0
1
0 6 15 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star4
0
1
0 6 16 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star5
0
1
0 6 17 5
1
end_operator
begin_operator
turn_to satellite1 groundstation7 star8
0
1
0 6 18 5
1
end_operator
begin_operator
turn_to satellite1 groundstation9 groundstation0
0
1
0 6 0 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 groundstation12
0
1
0 6 1 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 groundstation14
0
1
0 6 2 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 groundstation2
0
1
0 6 3 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 groundstation6
0
1
0 6 4 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 groundstation7
0
1
0 6 5 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 phenomenon16
0
1
0 6 7 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 planet15
0
1
0 6 8 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star1
0
1
0 6 9 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star10
0
1
0 6 10 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star11
0
1
0 6 11 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star13
0
1
0 6 12 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star17
0
1
0 6 13 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star18
0
1
0 6 14 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star3
0
1
0 6 15 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star4
0
1
0 6 16 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star5
0
1
0 6 17 6
1
end_operator
begin_operator
turn_to satellite1 groundstation9 star8
0
1
0 6 18 6
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 groundstation0
0
1
0 6 0 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 groundstation12
0
1
0 6 1 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 groundstation14
0
1
0 6 2 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 groundstation2
0
1
0 6 3 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 groundstation6
0
1
0 6 4 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 groundstation7
0
1
0 6 5 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 groundstation9
0
1
0 6 6 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 planet15
0
1
0 6 8 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 star1
0
1
0 6 9 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 star10
0
1
0 6 10 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 star11
0
1
0 6 11 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 star13
0
1
0 6 12 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 star17
0
1
0 6 13 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 star18
0
1
0 6 14 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 star3
0
1
0 6 15 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 star4
0
1
0 6 16 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 star5
0
1
0 6 17 7
1
end_operator
begin_operator
turn_to satellite1 phenomenon16 star8
0
1
0 6 18 7
1
end_operator
begin_operator
turn_to satellite1 planet15 groundstation0
0
1
0 6 0 8
1
end_operator
begin_operator
turn_to satellite1 planet15 groundstation12
0
1
0 6 1 8
1
end_operator
begin_operator
turn_to satellite1 planet15 groundstation14
0
1
0 6 2 8
1
end_operator
begin_operator
turn_to satellite1 planet15 groundstation2
0
1
0 6 3 8
1
end_operator
begin_operator
turn_to satellite1 planet15 groundstation6
0
1
0 6 4 8
1
end_operator
begin_operator
turn_to satellite1 planet15 groundstation7
0
1
0 6 5 8
1
end_operator
begin_operator
turn_to satellite1 planet15 groundstation9
0
1
0 6 6 8
1
end_operator
begin_operator
turn_to satellite1 planet15 phenomenon16
0
1
0 6 7 8
1
end_operator
begin_operator
turn_to satellite1 planet15 star1
0
1
0 6 9 8
1
end_operator
begin_operator
turn_to satellite1 planet15 star10
0
1
0 6 10 8
1
end_operator
begin_operator
turn_to satellite1 planet15 star11
0
1
0 6 11 8
1
end_operator
begin_operator
turn_to satellite1 planet15 star13
0
1
0 6 12 8
1
end_operator
begin_operator
turn_to satellite1 planet15 star17
0
1
0 6 13 8
1
end_operator
begin_operator
turn_to satellite1 planet15 star18
0
1
0 6 14 8
1
end_operator
begin_operator
turn_to satellite1 planet15 star3
0
1
0 6 15 8
1
end_operator
begin_operator
turn_to satellite1 planet15 star4
0
1
0 6 16 8
1
end_operator
begin_operator
turn_to satellite1 planet15 star5
0
1
0 6 17 8
1
end_operator
begin_operator
turn_to satellite1 planet15 star8
0
1
0 6 18 8
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation0
0
1
0 6 0 9
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation12
0
1
0 6 1 9
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation14
0
1
0 6 2 9
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation2
0
1
0 6 3 9
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation6
0
1
0 6 4 9
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation7
0
1
0 6 5 9
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation9
0
1
0 6 6 9
1
end_operator
begin_operator
turn_to satellite1 star1 phenomenon16
0
1
0 6 7 9
1
end_operator
begin_operator
turn_to satellite1 star1 planet15
0
1
0 6 8 9
1
end_operator
begin_operator
turn_to satellite1 star1 star10
0
1
0 6 10 9
1
end_operator
begin_operator
turn_to satellite1 star1 star11
0
1
0 6 11 9
1
end_operator
begin_operator
turn_to satellite1 star1 star13
0
1
0 6 12 9
1
end_operator
begin_operator
turn_to satellite1 star1 star17
0
1
0 6 13 9
1
end_operator
begin_operator
turn_to satellite1 star1 star18
0
1
0 6 14 9
1
end_operator
begin_operator
turn_to satellite1 star1 star3
0
1
0 6 15 9
1
end_operator
begin_operator
turn_to satellite1 star1 star4
0
1
0 6 16 9
1
end_operator
begin_operator
turn_to satellite1 star1 star5
0
1
0 6 17 9
1
end_operator
begin_operator
turn_to satellite1 star1 star8
0
1
0 6 18 9
1
end_operator
begin_operator
turn_to satellite1 star10 groundstation0
0
1
0 6 0 10
1
end_operator
begin_operator
turn_to satellite1 star10 groundstation12
0
1
0 6 1 10
1
end_operator
begin_operator
turn_to satellite1 star10 groundstation14
0
1
0 6 2 10
1
end_operator
begin_operator
turn_to satellite1 star10 groundstation2
0
1
0 6 3 10
1
end_operator
begin_operator
turn_to satellite1 star10 groundstation6
0
1
0 6 4 10
1
end_operator
begin_operator
turn_to satellite1 star10 groundstation7
0
1
0 6 5 10
1
end_operator
begin_operator
turn_to satellite1 star10 groundstation9
0
1
0 6 6 10
1
end_operator
begin_operator
turn_to satellite1 star10 phenomenon16
0
1
0 6 7 10
1
end_operator
begin_operator
turn_to satellite1 star10 planet15
0
1
0 6 8 10
1
end_operator
begin_operator
turn_to satellite1 star10 star1
0
1
0 6 9 10
1
end_operator
begin_operator
turn_to satellite1 star10 star11
0
1
0 6 11 10
1
end_operator
begin_operator
turn_to satellite1 star10 star13
0
1
0 6 12 10
1
end_operator
begin_operator
turn_to satellite1 star10 star17
0
1
0 6 13 10
1
end_operator
begin_operator
turn_to satellite1 star10 star18
0
1
0 6 14 10
1
end_operator
begin_operator
turn_to satellite1 star10 star3
0
1
0 6 15 10
1
end_operator
begin_operator
turn_to satellite1 star10 star4
0
1
0 6 16 10
1
end_operator
begin_operator
turn_to satellite1 star10 star5
0
1
0 6 17 10
1
end_operator
begin_operator
turn_to satellite1 star10 star8
0
1
0 6 18 10
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation0
0
1
0 6 0 11
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation12
0
1
0 6 1 11
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation14
0
1
0 6 2 11
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation2
0
1
0 6 3 11
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation6
0
1
0 6 4 11
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation7
0
1
0 6 5 11
1
end_operator
begin_operator
turn_to satellite1 star11 groundstation9
0
1
0 6 6 11
1
end_operator
begin_operator
turn_to satellite1 star11 phenomenon16
0
1
0 6 7 11
1
end_operator
begin_operator
turn_to satellite1 star11 planet15
0
1
0 6 8 11
1
end_operator
begin_operator
turn_to satellite1 star11 star1
0
1
0 6 9 11
1
end_operator
begin_operator
turn_to satellite1 star11 star10
0
1
0 6 10 11
1
end_operator
begin_operator
turn_to satellite1 star11 star13
0
1
0 6 12 11
1
end_operator
begin_operator
turn_to satellite1 star11 star17
0
1
0 6 13 11
1
end_operator
begin_operator
turn_to satellite1 star11 star18
0
1
0 6 14 11
1
end_operator
begin_operator
turn_to satellite1 star11 star3
0
1
0 6 15 11
1
end_operator
begin_operator
turn_to satellite1 star11 star4
0
1
0 6 16 11
1
end_operator
begin_operator
turn_to satellite1 star11 star5
0
1
0 6 17 11
1
end_operator
begin_operator
turn_to satellite1 star11 star8
0
1
0 6 18 11
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation0
0
1
0 6 0 12
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation12
0
1
0 6 1 12
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation14
0
1
0 6 2 12
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation2
0
1
0 6 3 12
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation6
0
1
0 6 4 12
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation7
0
1
0 6 5 12
1
end_operator
begin_operator
turn_to satellite1 star13 groundstation9
0
1
0 6 6 12
1
end_operator
begin_operator
turn_to satellite1 star13 phenomenon16
0
1
0 6 7 12
1
end_operator
begin_operator
turn_to satellite1 star13 planet15
0
1
0 6 8 12
1
end_operator
begin_operator
turn_to satellite1 star13 star1
0
1
0 6 9 12
1
end_operator
begin_operator
turn_to satellite1 star13 star10
0
1
0 6 10 12
1
end_operator
begin_operator
turn_to satellite1 star13 star11
0
1
0 6 11 12
1
end_operator
begin_operator
turn_to satellite1 star13 star17
0
1
0 6 13 12
1
end_operator
begin_operator
turn_to satellite1 star13 star18
0
1
0 6 14 12
1
end_operator
begin_operator
turn_to satellite1 star13 star3
0
1
0 6 15 12
1
end_operator
begin_operator
turn_to satellite1 star13 star4
0
1
0 6 16 12
1
end_operator
begin_operator
turn_to satellite1 star13 star5
0
1
0 6 17 12
1
end_operator
begin_operator
turn_to satellite1 star13 star8
0
1
0 6 18 12
1
end_operator
begin_operator
turn_to satellite1 star17 groundstation0
0
1
0 6 0 13
1
end_operator
begin_operator
turn_to satellite1 star17 groundstation12
0
1
0 6 1 13
1
end_operator
begin_operator
turn_to satellite1 star17 groundstation14
0
1
0 6 2 13
1
end_operator
begin_operator
turn_to satellite1 star17 groundstation2
0
1
0 6 3 13
1
end_operator
begin_operator
turn_to satellite1 star17 groundstation6
0
1
0 6 4 13
1
end_operator
begin_operator
turn_to satellite1 star17 groundstation7
0
1
0 6 5 13
1
end_operator
begin_operator
turn_to satellite1 star17 groundstation9
0
1
0 6 6 13
1
end_operator
begin_operator
turn_to satellite1 star17 phenomenon16
0
1
0 6 7 13
1
end_operator
begin_operator
turn_to satellite1 star17 planet15
0
1
0 6 8 13
1
end_operator
begin_operator
turn_to satellite1 star17 star1
0
1
0 6 9 13
1
end_operator
begin_operator
turn_to satellite1 star17 star10
0
1
0 6 10 13
1
end_operator
begin_operator
turn_to satellite1 star17 star11
0
1
0 6 11 13
1
end_operator
begin_operator
turn_to satellite1 star17 star13
0
1
0 6 12 13
1
end_operator
begin_operator
turn_to satellite1 star17 star18
0
1
0 6 14 13
1
end_operator
begin_operator
turn_to satellite1 star17 star3
0
1
0 6 15 13
1
end_operator
begin_operator
turn_to satellite1 star17 star4
0
1
0 6 16 13
1
end_operator
begin_operator
turn_to satellite1 star17 star5
0
1
0 6 17 13
1
end_operator
begin_operator
turn_to satellite1 star17 star8
0
1
0 6 18 13
1
end_operator
begin_operator
turn_to satellite1 star18 groundstation0
0
1
0 6 0 14
1
end_operator
begin_operator
turn_to satellite1 star18 groundstation12
0
1
0 6 1 14
1
end_operator
begin_operator
turn_to satellite1 star18 groundstation14
0
1
0 6 2 14
1
end_operator
begin_operator
turn_to satellite1 star18 groundstation2
0
1
0 6 3 14
1
end_operator
begin_operator
turn_to satellite1 star18 groundstation6
0
1
0 6 4 14
1
end_operator
begin_operator
turn_to satellite1 star18 groundstation7
0
1
0 6 5 14
1
end_operator
begin_operator
turn_to satellite1 star18 groundstation9
0
1
0 6 6 14
1
end_operator
begin_operator
turn_to satellite1 star18 phenomenon16
0
1
0 6 7 14
1
end_operator
begin_operator
turn_to satellite1 star18 planet15
0
1
0 6 8 14
1
end_operator
begin_operator
turn_to satellite1 star18 star1
0
1
0 6 9 14
1
end_operator
begin_operator
turn_to satellite1 star18 star10
0
1
0 6 10 14
1
end_operator
begin_operator
turn_to satellite1 star18 star11
0
1
0 6 11 14
1
end_operator
begin_operator
turn_to satellite1 star18 star13
0
1
0 6 12 14
1
end_operator
begin_operator
turn_to satellite1 star18 star17
0
1
0 6 13 14
1
end_operator
begin_operator
turn_to satellite1 star18 star3
0
1
0 6 15 14
1
end_operator
begin_operator
turn_to satellite1 star18 star4
0
1
0 6 16 14
1
end_operator
begin_operator
turn_to satellite1 star18 star5
0
1
0 6 17 14
1
end_operator
begin_operator
turn_to satellite1 star18 star8
0
1
0 6 18 14
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation0
0
1
0 6 0 15
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation12
0
1
0 6 1 15
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation14
0
1
0 6 2 15
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation2
0
1
0 6 3 15
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation6
0
1
0 6 4 15
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation7
0
1
0 6 5 15
1
end_operator
begin_operator
turn_to satellite1 star3 groundstation9
0
1
0 6 6 15
1
end_operator
begin_operator
turn_to satellite1 star3 phenomenon16
0
1
0 6 7 15
1
end_operator
begin_operator
turn_to satellite1 star3 planet15
0
1
0 6 8 15
1
end_operator
begin_operator
turn_to satellite1 star3 star1
0
1
0 6 9 15
1
end_operator
begin_operator
turn_to satellite1 star3 star10
0
1
0 6 10 15
1
end_operator
begin_operator
turn_to satellite1 star3 star11
0
1
0 6 11 15
1
end_operator
begin_operator
turn_to satellite1 star3 star13
0
1
0 6 12 15
1
end_operator
begin_operator
turn_to satellite1 star3 star17
0
1
0 6 13 15
1
end_operator
begin_operator
turn_to satellite1 star3 star18
0
1
0 6 14 15
1
end_operator
begin_operator
turn_to satellite1 star3 star4
0
1
0 6 16 15
1
end_operator
begin_operator
turn_to satellite1 star3 star5
0
1
0 6 17 15
1
end_operator
begin_operator
turn_to satellite1 star3 star8
0
1
0 6 18 15
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation0
0
1
0 6 0 16
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation12
0
1
0 6 1 16
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation14
0
1
0 6 2 16
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation2
0
1
0 6 3 16
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation6
0
1
0 6 4 16
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation7
0
1
0 6 5 16
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation9
0
1
0 6 6 16
1
end_operator
begin_operator
turn_to satellite1 star4 phenomenon16
0
1
0 6 7 16
1
end_operator
begin_operator
turn_to satellite1 star4 planet15
0
1
0 6 8 16
1
end_operator
begin_operator
turn_to satellite1 star4 star1
0
1
0 6 9 16
1
end_operator
begin_operator
turn_to satellite1 star4 star10
0
1
0 6 10 16
1
end_operator
begin_operator
turn_to satellite1 star4 star11
0
1
0 6 11 16
1
end_operator
begin_operator
turn_to satellite1 star4 star13
0
1
0 6 12 16
1
end_operator
begin_operator
turn_to satellite1 star4 star17
0
1
0 6 13 16
1
end_operator
begin_operator
turn_to satellite1 star4 star18
0
1
0 6 14 16
1
end_operator
begin_operator
turn_to satellite1 star4 star3
0
1
0 6 15 16
1
end_operator
begin_operator
turn_to satellite1 star4 star5
0
1
0 6 17 16
1
end_operator
begin_operator
turn_to satellite1 star4 star8
0
1
0 6 18 16
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation0
0
1
0 6 0 17
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation12
0
1
0 6 1 17
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation14
0
1
0 6 2 17
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation2
0
1
0 6 3 17
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation6
0
1
0 6 4 17
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation7
0
1
0 6 5 17
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation9
0
1
0 6 6 17
1
end_operator
begin_operator
turn_to satellite1 star5 phenomenon16
0
1
0 6 7 17
1
end_operator
begin_operator
turn_to satellite1 star5 planet15
0
1
0 6 8 17
1
end_operator
begin_operator
turn_to satellite1 star5 star1
0
1
0 6 9 17
1
end_operator
begin_operator
turn_to satellite1 star5 star10
0
1
0 6 10 17
1
end_operator
begin_operator
turn_to satellite1 star5 star11
0
1
0 6 11 17
1
end_operator
begin_operator
turn_to satellite1 star5 star13
0
1
0 6 12 17
1
end_operator
begin_operator
turn_to satellite1 star5 star17
0
1
0 6 13 17
1
end_operator
begin_operator
turn_to satellite1 star5 star18
0
1
0 6 14 17
1
end_operator
begin_operator
turn_to satellite1 star5 star3
0
1
0 6 15 17
1
end_operator
begin_operator
turn_to satellite1 star5 star4
0
1
0 6 16 17
1
end_operator
begin_operator
turn_to satellite1 star5 star8
0
1
0 6 18 17
1
end_operator
begin_operator
turn_to satellite1 star8 groundstation0
0
1
0 6 0 18
1
end_operator
begin_operator
turn_to satellite1 star8 groundstation12
0
1
0 6 1 18
1
end_operator
begin_operator
turn_to satellite1 star8 groundstation14
0
1
0 6 2 18
1
end_operator
begin_operator
turn_to satellite1 star8 groundstation2
0
1
0 6 3 18
1
end_operator
begin_operator
turn_to satellite1 star8 groundstation6
0
1
0 6 4 18
1
end_operator
begin_operator
turn_to satellite1 star8 groundstation7
0
1
0 6 5 18
1
end_operator
begin_operator
turn_to satellite1 star8 groundstation9
0
1
0 6 6 18
1
end_operator
begin_operator
turn_to satellite1 star8 phenomenon16
0
1
0 6 7 18
1
end_operator
begin_operator
turn_to satellite1 star8 planet15
0
1
0 6 8 18
1
end_operator
begin_operator
turn_to satellite1 star8 star1
0
1
0 6 9 18
1
end_operator
begin_operator
turn_to satellite1 star8 star10
0
1
0 6 10 18
1
end_operator
begin_operator
turn_to satellite1 star8 star11
0
1
0 6 11 18
1
end_operator
begin_operator
turn_to satellite1 star8 star13
0
1
0 6 12 18
1
end_operator
begin_operator
turn_to satellite1 star8 star17
0
1
0 6 13 18
1
end_operator
begin_operator
turn_to satellite1 star8 star18
0
1
0 6 14 18
1
end_operator
begin_operator
turn_to satellite1 star8 star3
0
1
0 6 15 18
1
end_operator
begin_operator
turn_to satellite1 star8 star4
0
1
0 6 16 18
1
end_operator
begin_operator
turn_to satellite1 star8 star5
0
1
0 6 17 18
1
end_operator
0
