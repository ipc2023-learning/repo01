begin_version
3
end_version
begin_metric
0
end_metric
12
begin_variable
var0
-1
2
Atom power_on(instrument4)
NegatedAtom power_on(instrument4)
end_variable
begin_variable
var1
-1
2
Atom power_on(instrument5)
NegatedAtom power_on(instrument5)
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
Atom power_on(instrument3)
NegatedAtom power_on(instrument3)
end_variable
begin_variable
var5
-1
14
Atom pointing(satellite3, groundstation0)
Atom pointing(satellite3, groundstation2)
Atom pointing(satellite3, groundstation6)
Atom pointing(satellite3, groundstation7)
Atom pointing(satellite3, groundstation9)
Atom pointing(satellite3, phenomenon12)
Atom pointing(satellite3, phenomenon13)
Atom pointing(satellite3, planet10)
Atom pointing(satellite3, star1)
Atom pointing(satellite3, star11)
Atom pointing(satellite3, star3)
Atom pointing(satellite3, star4)
Atom pointing(satellite3, star5)
Atom pointing(satellite3, star8)
end_variable
begin_variable
var6
-1
14
Atom pointing(satellite2, groundstation0)
Atom pointing(satellite2, groundstation2)
Atom pointing(satellite2, groundstation6)
Atom pointing(satellite2, groundstation7)
Atom pointing(satellite2, groundstation9)
Atom pointing(satellite2, phenomenon12)
Atom pointing(satellite2, phenomenon13)
Atom pointing(satellite2, planet10)
Atom pointing(satellite2, star1)
Atom pointing(satellite2, star11)
Atom pointing(satellite2, star3)
Atom pointing(satellite2, star4)
Atom pointing(satellite2, star5)
Atom pointing(satellite2, star8)
end_variable
begin_variable
var7
-1
2
Atom calibrated(instrument5)
NegatedAtom calibrated(instrument5)
end_variable
begin_variable
var8
-1
2
Atom calibrated(instrument4)
NegatedAtom calibrated(instrument4)
end_variable
begin_variable
var9
-1
2
Atom calibrated(instrument3)
NegatedAtom calibrated(instrument3)
end_variable
begin_variable
var10
-1
2
Atom have_image(star11, spectrograph3)
NegatedAtom have_image(star11, spectrograph3)
end_variable
begin_variable
var11
-1
2
Atom have_image(phenomenon13, spectrograph3)
NegatedAtom have_image(phenomenon13, spectrograph3)
end_variable
0
begin_state
1
1
0
0
1
0
0
1
1
1
1
1
end_state
begin_goal
2
10 0
11 0
end_goal
383
begin_operator
calibrate satellite2 instrument3 groundstation7
2
6 3
4 0
1
0 9 -1 0
1
end_operator
begin_operator
calibrate satellite2 instrument3 star1
2
6 8
4 0
1
0 9 -1 0
1
end_operator
begin_operator
calibrate satellite2 instrument3 star5
2
6 12
4 0
1
0 9 -1 0
1
end_operator
begin_operator
calibrate satellite3 instrument4 groundstation2
2
5 1
0 0
1
0 8 -1 0
1
end_operator
begin_operator
calibrate satellite3 instrument4 star1
2
5 8
0 0
1
0 8 -1 0
1
end_operator
begin_operator
calibrate satellite3 instrument4 star8
2
5 13
0 0
1
0 8 -1 0
1
end_operator
begin_operator
calibrate satellite3 instrument5 star1
2
5 8
1 0
1
0 7 -1 0
1
end_operator
begin_operator
switch_off instrument3 satellite2
0
2
0 3 -1 0
0 4 0 1
1
end_operator
begin_operator
switch_off instrument4 satellite3
0
2
0 2 -1 0
0 0 0 1
1
end_operator
begin_operator
switch_off instrument5 satellite3
0
2
0 2 -1 0
0 1 0 1
1
end_operator
begin_operator
switch_on instrument3 satellite2
0
3
0 9 -1 1
0 3 0 1
0 4 -1 0
1
end_operator
begin_operator
switch_on instrument4 satellite3
0
3
0 8 -1 1
0 2 0 1
0 0 -1 0
1
end_operator
begin_operator
switch_on instrument5 satellite3
0
3
0 7 -1 1
0 2 0 1
0 1 -1 0
1
end_operator
begin_operator
take_image satellite2 phenomenon13 instrument3 spectrograph3
3
9 0
6 6
4 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite2 star11 instrument3 spectrograph3
3
9 0
6 9
4 0
1
0 10 -1 0
1
end_operator
begin_operator
take_image satellite3 phenomenon13 instrument4 spectrograph3
3
8 0
5 6
0 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite3 phenomenon13 instrument5 spectrograph3
3
7 0
5 6
1 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite3 star11 instrument4 spectrograph3
3
8 0
5 9
0 0
1
0 10 -1 0
1
end_operator
begin_operator
take_image satellite3 star11 instrument5 spectrograph3
3
7 0
5 9
1 0
1
0 10 -1 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 groundstation2
0
1
0 6 1 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 groundstation6
0
1
0 6 2 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 groundstation7
0
1
0 6 3 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 groundstation9
0
1
0 6 4 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 phenomenon12
0
1
0 6 5 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 phenomenon13
0
1
0 6 6 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 planet10
0
1
0 6 7 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 star1
0
1
0 6 8 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 star11
0
1
0 6 9 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 star3
0
1
0 6 10 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 star4
0
1
0 6 11 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 star5
0
1
0 6 12 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 star8
0
1
0 6 13 0
1
end_operator
begin_operator
turn_to satellite2 groundstation2 groundstation0
0
1
0 6 0 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 groundstation6
0
1
0 6 2 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 groundstation7
0
1
0 6 3 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 groundstation9
0
1
0 6 4 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 phenomenon12
0
1
0 6 5 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 phenomenon13
0
1
0 6 6 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 planet10
0
1
0 6 7 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 star1
0
1
0 6 8 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 star11
0
1
0 6 9 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 star3
0
1
0 6 10 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 star4
0
1
0 6 11 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 star5
0
1
0 6 12 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 star8
0
1
0 6 13 1
1
end_operator
begin_operator
turn_to satellite2 groundstation6 groundstation0
0
1
0 6 0 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 groundstation2
0
1
0 6 1 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 groundstation7
0
1
0 6 3 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 groundstation9
0
1
0 6 4 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 phenomenon12
0
1
0 6 5 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 phenomenon13
0
1
0 6 6 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 planet10
0
1
0 6 7 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 star1
0
1
0 6 8 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 star11
0
1
0 6 9 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 star3
0
1
0 6 10 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 star4
0
1
0 6 11 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 star5
0
1
0 6 12 2
1
end_operator
begin_operator
turn_to satellite2 groundstation6 star8
0
1
0 6 13 2
1
end_operator
begin_operator
turn_to satellite2 groundstation7 groundstation0
0
1
0 6 0 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 groundstation2
0
1
0 6 1 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 groundstation6
0
1
0 6 2 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 groundstation9
0
1
0 6 4 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 phenomenon12
0
1
0 6 5 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 phenomenon13
0
1
0 6 6 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 planet10
0
1
0 6 7 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star1
0
1
0 6 8 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star11
0
1
0 6 9 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star3
0
1
0 6 10 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star4
0
1
0 6 11 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star5
0
1
0 6 12 3
1
end_operator
begin_operator
turn_to satellite2 groundstation7 star8
0
1
0 6 13 3
1
end_operator
begin_operator
turn_to satellite2 groundstation9 groundstation0
0
1
0 6 0 4
1
end_operator
begin_operator
turn_to satellite2 groundstation9 groundstation2
0
1
0 6 1 4
1
end_operator
begin_operator
turn_to satellite2 groundstation9 groundstation6
0
1
0 6 2 4
1
end_operator
begin_operator
turn_to satellite2 groundstation9 groundstation7
0
1
0 6 3 4
1
end_operator
begin_operator
turn_to satellite2 groundstation9 phenomenon12
0
1
0 6 5 4
1
end_operator
begin_operator
turn_to satellite2 groundstation9 phenomenon13
0
1
0 6 6 4
1
end_operator
begin_operator
turn_to satellite2 groundstation9 planet10
0
1
0 6 7 4
1
end_operator
begin_operator
turn_to satellite2 groundstation9 star1
0
1
0 6 8 4
1
end_operator
begin_operator
turn_to satellite2 groundstation9 star11
0
1
0 6 9 4
1
end_operator
begin_operator
turn_to satellite2 groundstation9 star3
0
1
0 6 10 4
1
end_operator
begin_operator
turn_to satellite2 groundstation9 star4
0
1
0 6 11 4
1
end_operator
begin_operator
turn_to satellite2 groundstation9 star5
0
1
0 6 12 4
1
end_operator
begin_operator
turn_to satellite2 groundstation9 star8
0
1
0 6 13 4
1
end_operator
begin_operator
turn_to satellite2 phenomenon12 groundstation0
0
1
0 6 0 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon12 groundstation2
0
1
0 6 1 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon12 groundstation6
0
1
0 6 2 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon12 groundstation7
0
1
0 6 3 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon12 groundstation9
0
1
0 6 4 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon12 phenomenon13
0
1
0 6 6 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon12 planet10
0
1
0 6 7 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon12 star1
0
1
0 6 8 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon12 star11
0
1
0 6 9 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon12 star3
0
1
0 6 10 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon12 star4
0
1
0 6 11 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon12 star5
0
1
0 6 12 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon12 star8
0
1
0 6 13 5
1
end_operator
begin_operator
turn_to satellite2 phenomenon13 groundstation0
0
1
0 6 0 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon13 groundstation2
0
1
0 6 1 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon13 groundstation6
0
1
0 6 2 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon13 groundstation7
0
1
0 6 3 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon13 groundstation9
0
1
0 6 4 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon13 phenomenon12
0
1
0 6 5 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon13 planet10
0
1
0 6 7 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon13 star1
0
1
0 6 8 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon13 star11
0
1
0 6 9 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon13 star3
0
1
0 6 10 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon13 star4
0
1
0 6 11 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon13 star5
0
1
0 6 12 6
1
end_operator
begin_operator
turn_to satellite2 phenomenon13 star8
0
1
0 6 13 6
1
end_operator
begin_operator
turn_to satellite2 planet10 groundstation0
0
1
0 6 0 7
1
end_operator
begin_operator
turn_to satellite2 planet10 groundstation2
0
1
0 6 1 7
1
end_operator
begin_operator
turn_to satellite2 planet10 groundstation6
0
1
0 6 2 7
1
end_operator
begin_operator
turn_to satellite2 planet10 groundstation7
0
1
0 6 3 7
1
end_operator
begin_operator
turn_to satellite2 planet10 groundstation9
0
1
0 6 4 7
1
end_operator
begin_operator
turn_to satellite2 planet10 phenomenon12
0
1
0 6 5 7
1
end_operator
begin_operator
turn_to satellite2 planet10 phenomenon13
0
1
0 6 6 7
1
end_operator
begin_operator
turn_to satellite2 planet10 star1
0
1
0 6 8 7
1
end_operator
begin_operator
turn_to satellite2 planet10 star11
0
1
0 6 9 7
1
end_operator
begin_operator
turn_to satellite2 planet10 star3
0
1
0 6 10 7
1
end_operator
begin_operator
turn_to satellite2 planet10 star4
0
1
0 6 11 7
1
end_operator
begin_operator
turn_to satellite2 planet10 star5
0
1
0 6 12 7
1
end_operator
begin_operator
turn_to satellite2 planet10 star8
0
1
0 6 13 7
1
end_operator
begin_operator
turn_to satellite2 star1 groundstation0
0
1
0 6 0 8
1
end_operator
begin_operator
turn_to satellite2 star1 groundstation2
0
1
0 6 1 8
1
end_operator
begin_operator
turn_to satellite2 star1 groundstation6
0
1
0 6 2 8
1
end_operator
begin_operator
turn_to satellite2 star1 groundstation7
0
1
0 6 3 8
1
end_operator
begin_operator
turn_to satellite2 star1 groundstation9
0
1
0 6 4 8
1
end_operator
begin_operator
turn_to satellite2 star1 phenomenon12
0
1
0 6 5 8
1
end_operator
begin_operator
turn_to satellite2 star1 phenomenon13
0
1
0 6 6 8
1
end_operator
begin_operator
turn_to satellite2 star1 planet10
0
1
0 6 7 8
1
end_operator
begin_operator
turn_to satellite2 star1 star11
0
1
0 6 9 8
1
end_operator
begin_operator
turn_to satellite2 star1 star3
0
1
0 6 10 8
1
end_operator
begin_operator
turn_to satellite2 star1 star4
0
1
0 6 11 8
1
end_operator
begin_operator
turn_to satellite2 star1 star5
0
1
0 6 12 8
1
end_operator
begin_operator
turn_to satellite2 star1 star8
0
1
0 6 13 8
1
end_operator
begin_operator
turn_to satellite2 star11 groundstation0
0
1
0 6 0 9
1
end_operator
begin_operator
turn_to satellite2 star11 groundstation2
0
1
0 6 1 9
1
end_operator
begin_operator
turn_to satellite2 star11 groundstation6
0
1
0 6 2 9
1
end_operator
begin_operator
turn_to satellite2 star11 groundstation7
0
1
0 6 3 9
1
end_operator
begin_operator
turn_to satellite2 star11 groundstation9
0
1
0 6 4 9
1
end_operator
begin_operator
turn_to satellite2 star11 phenomenon12
0
1
0 6 5 9
1
end_operator
begin_operator
turn_to satellite2 star11 phenomenon13
0
1
0 6 6 9
1
end_operator
begin_operator
turn_to satellite2 star11 planet10
0
1
0 6 7 9
1
end_operator
begin_operator
turn_to satellite2 star11 star1
0
1
0 6 8 9
1
end_operator
begin_operator
turn_to satellite2 star11 star3
0
1
0 6 10 9
1
end_operator
begin_operator
turn_to satellite2 star11 star4
0
1
0 6 11 9
1
end_operator
begin_operator
turn_to satellite2 star11 star5
0
1
0 6 12 9
1
end_operator
begin_operator
turn_to satellite2 star11 star8
0
1
0 6 13 9
1
end_operator
begin_operator
turn_to satellite2 star3 groundstation0
0
1
0 6 0 10
1
end_operator
begin_operator
turn_to satellite2 star3 groundstation2
0
1
0 6 1 10
1
end_operator
begin_operator
turn_to satellite2 star3 groundstation6
0
1
0 6 2 10
1
end_operator
begin_operator
turn_to satellite2 star3 groundstation7
0
1
0 6 3 10
1
end_operator
begin_operator
turn_to satellite2 star3 groundstation9
0
1
0 6 4 10
1
end_operator
begin_operator
turn_to satellite2 star3 phenomenon12
0
1
0 6 5 10
1
end_operator
begin_operator
turn_to satellite2 star3 phenomenon13
0
1
0 6 6 10
1
end_operator
begin_operator
turn_to satellite2 star3 planet10
0
1
0 6 7 10
1
end_operator
begin_operator
turn_to satellite2 star3 star1
0
1
0 6 8 10
1
end_operator
begin_operator
turn_to satellite2 star3 star11
0
1
0 6 9 10
1
end_operator
begin_operator
turn_to satellite2 star3 star4
0
1
0 6 11 10
1
end_operator
begin_operator
turn_to satellite2 star3 star5
0
1
0 6 12 10
1
end_operator
begin_operator
turn_to satellite2 star3 star8
0
1
0 6 13 10
1
end_operator
begin_operator
turn_to satellite2 star4 groundstation0
0
1
0 6 0 11
1
end_operator
begin_operator
turn_to satellite2 star4 groundstation2
0
1
0 6 1 11
1
end_operator
begin_operator
turn_to satellite2 star4 groundstation6
0
1
0 6 2 11
1
end_operator
begin_operator
turn_to satellite2 star4 groundstation7
0
1
0 6 3 11
1
end_operator
begin_operator
turn_to satellite2 star4 groundstation9
0
1
0 6 4 11
1
end_operator
begin_operator
turn_to satellite2 star4 phenomenon12
0
1
0 6 5 11
1
end_operator
begin_operator
turn_to satellite2 star4 phenomenon13
0
1
0 6 6 11
1
end_operator
begin_operator
turn_to satellite2 star4 planet10
0
1
0 6 7 11
1
end_operator
begin_operator
turn_to satellite2 star4 star1
0
1
0 6 8 11
1
end_operator
begin_operator
turn_to satellite2 star4 star11
0
1
0 6 9 11
1
end_operator
begin_operator
turn_to satellite2 star4 star3
0
1
0 6 10 11
1
end_operator
begin_operator
turn_to satellite2 star4 star5
0
1
0 6 12 11
1
end_operator
begin_operator
turn_to satellite2 star4 star8
0
1
0 6 13 11
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation0
0
1
0 6 0 12
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation2
0
1
0 6 1 12
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation6
0
1
0 6 2 12
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation7
0
1
0 6 3 12
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation9
0
1
0 6 4 12
1
end_operator
begin_operator
turn_to satellite2 star5 phenomenon12
0
1
0 6 5 12
1
end_operator
begin_operator
turn_to satellite2 star5 phenomenon13
0
1
0 6 6 12
1
end_operator
begin_operator
turn_to satellite2 star5 planet10
0
1
0 6 7 12
1
end_operator
begin_operator
turn_to satellite2 star5 star1
0
1
0 6 8 12
1
end_operator
begin_operator
turn_to satellite2 star5 star11
0
1
0 6 9 12
1
end_operator
begin_operator
turn_to satellite2 star5 star3
0
1
0 6 10 12
1
end_operator
begin_operator
turn_to satellite2 star5 star4
0
1
0 6 11 12
1
end_operator
begin_operator
turn_to satellite2 star5 star8
0
1
0 6 13 12
1
end_operator
begin_operator
turn_to satellite2 star8 groundstation0
0
1
0 6 0 13
1
end_operator
begin_operator
turn_to satellite2 star8 groundstation2
0
1
0 6 1 13
1
end_operator
begin_operator
turn_to satellite2 star8 groundstation6
0
1
0 6 2 13
1
end_operator
begin_operator
turn_to satellite2 star8 groundstation7
0
1
0 6 3 13
1
end_operator
begin_operator
turn_to satellite2 star8 groundstation9
0
1
0 6 4 13
1
end_operator
begin_operator
turn_to satellite2 star8 phenomenon12
0
1
0 6 5 13
1
end_operator
begin_operator
turn_to satellite2 star8 phenomenon13
0
1
0 6 6 13
1
end_operator
begin_operator
turn_to satellite2 star8 planet10
0
1
0 6 7 13
1
end_operator
begin_operator
turn_to satellite2 star8 star1
0
1
0 6 8 13
1
end_operator
begin_operator
turn_to satellite2 star8 star11
0
1
0 6 9 13
1
end_operator
begin_operator
turn_to satellite2 star8 star3
0
1
0 6 10 13
1
end_operator
begin_operator
turn_to satellite2 star8 star4
0
1
0 6 11 13
1
end_operator
begin_operator
turn_to satellite2 star8 star5
0
1
0 6 12 13
1
end_operator
begin_operator
turn_to satellite3 groundstation0 groundstation2
0
1
0 5 1 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 groundstation6
0
1
0 5 2 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 groundstation7
0
1
0 5 3 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 groundstation9
0
1
0 5 4 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 phenomenon12
0
1
0 5 5 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 phenomenon13
0
1
0 5 6 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 planet10
0
1
0 5 7 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 star1
0
1
0 5 8 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 star11
0
1
0 5 9 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 star3
0
1
0 5 10 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 star4
0
1
0 5 11 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 star5
0
1
0 5 12 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 star8
0
1
0 5 13 0
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation0
0
1
0 5 0 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation6
0
1
0 5 2 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation7
0
1
0 5 3 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation9
0
1
0 5 4 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 phenomenon12
0
1
0 5 5 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 phenomenon13
0
1
0 5 6 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 planet10
0
1
0 5 7 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star1
0
1
0 5 8 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star11
0
1
0 5 9 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star3
0
1
0 5 10 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star4
0
1
0 5 11 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star5
0
1
0 5 12 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star8
0
1
0 5 13 1
1
end_operator
begin_operator
turn_to satellite3 groundstation6 groundstation0
0
1
0 5 0 2
1
end_operator
begin_operator
turn_to satellite3 groundstation6 groundstation2
0
1
0 5 1 2
1
end_operator
begin_operator
turn_to satellite3 groundstation6 groundstation7
0
1
0 5 3 2
1
end_operator
begin_operator
turn_to satellite3 groundstation6 groundstation9
0
1
0 5 4 2
1
end_operator
begin_operator
turn_to satellite3 groundstation6 phenomenon12
0
1
0 5 5 2
1
end_operator
begin_operator
turn_to satellite3 groundstation6 phenomenon13
0
1
0 5 6 2
1
end_operator
begin_operator
turn_to satellite3 groundstation6 planet10
0
1
0 5 7 2
1
end_operator
begin_operator
turn_to satellite3 groundstation6 star1
0
1
0 5 8 2
1
end_operator
begin_operator
turn_to satellite3 groundstation6 star11
0
1
0 5 9 2
1
end_operator
begin_operator
turn_to satellite3 groundstation6 star3
0
1
0 5 10 2
1
end_operator
begin_operator
turn_to satellite3 groundstation6 star4
0
1
0 5 11 2
1
end_operator
begin_operator
turn_to satellite3 groundstation6 star5
0
1
0 5 12 2
1
end_operator
begin_operator
turn_to satellite3 groundstation6 star8
0
1
0 5 13 2
1
end_operator
begin_operator
turn_to satellite3 groundstation7 groundstation0
0
1
0 5 0 3
1
end_operator
begin_operator
turn_to satellite3 groundstation7 groundstation2
0
1
0 5 1 3
1
end_operator
begin_operator
turn_to satellite3 groundstation7 groundstation6
0
1
0 5 2 3
1
end_operator
begin_operator
turn_to satellite3 groundstation7 groundstation9
0
1
0 5 4 3
1
end_operator
begin_operator
turn_to satellite3 groundstation7 phenomenon12
0
1
0 5 5 3
1
end_operator
begin_operator
turn_to satellite3 groundstation7 phenomenon13
0
1
0 5 6 3
1
end_operator
begin_operator
turn_to satellite3 groundstation7 planet10
0
1
0 5 7 3
1
end_operator
begin_operator
turn_to satellite3 groundstation7 star1
0
1
0 5 8 3
1
end_operator
begin_operator
turn_to satellite3 groundstation7 star11
0
1
0 5 9 3
1
end_operator
begin_operator
turn_to satellite3 groundstation7 star3
0
1
0 5 10 3
1
end_operator
begin_operator
turn_to satellite3 groundstation7 star4
0
1
0 5 11 3
1
end_operator
begin_operator
turn_to satellite3 groundstation7 star5
0
1
0 5 12 3
1
end_operator
begin_operator
turn_to satellite3 groundstation7 star8
0
1
0 5 13 3
1
end_operator
begin_operator
turn_to satellite3 groundstation9 groundstation0
0
1
0 5 0 4
1
end_operator
begin_operator
turn_to satellite3 groundstation9 groundstation2
0
1
0 5 1 4
1
end_operator
begin_operator
turn_to satellite3 groundstation9 groundstation6
0
1
0 5 2 4
1
end_operator
begin_operator
turn_to satellite3 groundstation9 groundstation7
0
1
0 5 3 4
1
end_operator
begin_operator
turn_to satellite3 groundstation9 phenomenon12
0
1
0 5 5 4
1
end_operator
begin_operator
turn_to satellite3 groundstation9 phenomenon13
0
1
0 5 6 4
1
end_operator
begin_operator
turn_to satellite3 groundstation9 planet10
0
1
0 5 7 4
1
end_operator
begin_operator
turn_to satellite3 groundstation9 star1
0
1
0 5 8 4
1
end_operator
begin_operator
turn_to satellite3 groundstation9 star11
0
1
0 5 9 4
1
end_operator
begin_operator
turn_to satellite3 groundstation9 star3
0
1
0 5 10 4
1
end_operator
begin_operator
turn_to satellite3 groundstation9 star4
0
1
0 5 11 4
1
end_operator
begin_operator
turn_to satellite3 groundstation9 star5
0
1
0 5 12 4
1
end_operator
begin_operator
turn_to satellite3 groundstation9 star8
0
1
0 5 13 4
1
end_operator
begin_operator
turn_to satellite3 phenomenon12 groundstation0
0
1
0 5 0 5
1
end_operator
begin_operator
turn_to satellite3 phenomenon12 groundstation2
0
1
0 5 1 5
1
end_operator
begin_operator
turn_to satellite3 phenomenon12 groundstation6
0
1
0 5 2 5
1
end_operator
begin_operator
turn_to satellite3 phenomenon12 groundstation7
0
1
0 5 3 5
1
end_operator
begin_operator
turn_to satellite3 phenomenon12 groundstation9
0
1
0 5 4 5
1
end_operator
begin_operator
turn_to satellite3 phenomenon12 phenomenon13
0
1
0 5 6 5
1
end_operator
begin_operator
turn_to satellite3 phenomenon12 planet10
0
1
0 5 7 5
1
end_operator
begin_operator
turn_to satellite3 phenomenon12 star1
0
1
0 5 8 5
1
end_operator
begin_operator
turn_to satellite3 phenomenon12 star11
0
1
0 5 9 5
1
end_operator
begin_operator
turn_to satellite3 phenomenon12 star3
0
1
0 5 10 5
1
end_operator
begin_operator
turn_to satellite3 phenomenon12 star4
0
1
0 5 11 5
1
end_operator
begin_operator
turn_to satellite3 phenomenon12 star5
0
1
0 5 12 5
1
end_operator
begin_operator
turn_to satellite3 phenomenon12 star8
0
1
0 5 13 5
1
end_operator
begin_operator
turn_to satellite3 phenomenon13 groundstation0
0
1
0 5 0 6
1
end_operator
begin_operator
turn_to satellite3 phenomenon13 groundstation2
0
1
0 5 1 6
1
end_operator
begin_operator
turn_to satellite3 phenomenon13 groundstation6
0
1
0 5 2 6
1
end_operator
begin_operator
turn_to satellite3 phenomenon13 groundstation7
0
1
0 5 3 6
1
end_operator
begin_operator
turn_to satellite3 phenomenon13 groundstation9
0
1
0 5 4 6
1
end_operator
begin_operator
turn_to satellite3 phenomenon13 phenomenon12
0
1
0 5 5 6
1
end_operator
begin_operator
turn_to satellite3 phenomenon13 planet10
0
1
0 5 7 6
1
end_operator
begin_operator
turn_to satellite3 phenomenon13 star1
0
1
0 5 8 6
1
end_operator
begin_operator
turn_to satellite3 phenomenon13 star11
0
1
0 5 9 6
1
end_operator
begin_operator
turn_to satellite3 phenomenon13 star3
0
1
0 5 10 6
1
end_operator
begin_operator
turn_to satellite3 phenomenon13 star4
0
1
0 5 11 6
1
end_operator
begin_operator
turn_to satellite3 phenomenon13 star5
0
1
0 5 12 6
1
end_operator
begin_operator
turn_to satellite3 phenomenon13 star8
0
1
0 5 13 6
1
end_operator
begin_operator
turn_to satellite3 planet10 groundstation0
0
1
0 5 0 7
1
end_operator
begin_operator
turn_to satellite3 planet10 groundstation2
0
1
0 5 1 7
1
end_operator
begin_operator
turn_to satellite3 planet10 groundstation6
0
1
0 5 2 7
1
end_operator
begin_operator
turn_to satellite3 planet10 groundstation7
0
1
0 5 3 7
1
end_operator
begin_operator
turn_to satellite3 planet10 groundstation9
0
1
0 5 4 7
1
end_operator
begin_operator
turn_to satellite3 planet10 phenomenon12
0
1
0 5 5 7
1
end_operator
begin_operator
turn_to satellite3 planet10 phenomenon13
0
1
0 5 6 7
1
end_operator
begin_operator
turn_to satellite3 planet10 star1
0
1
0 5 8 7
1
end_operator
begin_operator
turn_to satellite3 planet10 star11
0
1
0 5 9 7
1
end_operator
begin_operator
turn_to satellite3 planet10 star3
0
1
0 5 10 7
1
end_operator
begin_operator
turn_to satellite3 planet10 star4
0
1
0 5 11 7
1
end_operator
begin_operator
turn_to satellite3 planet10 star5
0
1
0 5 12 7
1
end_operator
begin_operator
turn_to satellite3 planet10 star8
0
1
0 5 13 7
1
end_operator
begin_operator
turn_to satellite3 star1 groundstation0
0
1
0 5 0 8
1
end_operator
begin_operator
turn_to satellite3 star1 groundstation2
0
1
0 5 1 8
1
end_operator
begin_operator
turn_to satellite3 star1 groundstation6
0
1
0 5 2 8
1
end_operator
begin_operator
turn_to satellite3 star1 groundstation7
0
1
0 5 3 8
1
end_operator
begin_operator
turn_to satellite3 star1 groundstation9
0
1
0 5 4 8
1
end_operator
begin_operator
turn_to satellite3 star1 phenomenon12
0
1
0 5 5 8
1
end_operator
begin_operator
turn_to satellite3 star1 phenomenon13
0
1
0 5 6 8
1
end_operator
begin_operator
turn_to satellite3 star1 planet10
0
1
0 5 7 8
1
end_operator
begin_operator
turn_to satellite3 star1 star11
0
1
0 5 9 8
1
end_operator
begin_operator
turn_to satellite3 star1 star3
0
1
0 5 10 8
1
end_operator
begin_operator
turn_to satellite3 star1 star4
0
1
0 5 11 8
1
end_operator
begin_operator
turn_to satellite3 star1 star5
0
1
0 5 12 8
1
end_operator
begin_operator
turn_to satellite3 star1 star8
0
1
0 5 13 8
1
end_operator
begin_operator
turn_to satellite3 star11 groundstation0
0
1
0 5 0 9
1
end_operator
begin_operator
turn_to satellite3 star11 groundstation2
0
1
0 5 1 9
1
end_operator
begin_operator
turn_to satellite3 star11 groundstation6
0
1
0 5 2 9
1
end_operator
begin_operator
turn_to satellite3 star11 groundstation7
0
1
0 5 3 9
1
end_operator
begin_operator
turn_to satellite3 star11 groundstation9
0
1
0 5 4 9
1
end_operator
begin_operator
turn_to satellite3 star11 phenomenon12
0
1
0 5 5 9
1
end_operator
begin_operator
turn_to satellite3 star11 phenomenon13
0
1
0 5 6 9
1
end_operator
begin_operator
turn_to satellite3 star11 planet10
0
1
0 5 7 9
1
end_operator
begin_operator
turn_to satellite3 star11 star1
0
1
0 5 8 9
1
end_operator
begin_operator
turn_to satellite3 star11 star3
0
1
0 5 10 9
1
end_operator
begin_operator
turn_to satellite3 star11 star4
0
1
0 5 11 9
1
end_operator
begin_operator
turn_to satellite3 star11 star5
0
1
0 5 12 9
1
end_operator
begin_operator
turn_to satellite3 star11 star8
0
1
0 5 13 9
1
end_operator
begin_operator
turn_to satellite3 star3 groundstation0
0
1
0 5 0 10
1
end_operator
begin_operator
turn_to satellite3 star3 groundstation2
0
1
0 5 1 10
1
end_operator
begin_operator
turn_to satellite3 star3 groundstation6
0
1
0 5 2 10
1
end_operator
begin_operator
turn_to satellite3 star3 groundstation7
0
1
0 5 3 10
1
end_operator
begin_operator
turn_to satellite3 star3 groundstation9
0
1
0 5 4 10
1
end_operator
begin_operator
turn_to satellite3 star3 phenomenon12
0
1
0 5 5 10
1
end_operator
begin_operator
turn_to satellite3 star3 phenomenon13
0
1
0 5 6 10
1
end_operator
begin_operator
turn_to satellite3 star3 planet10
0
1
0 5 7 10
1
end_operator
begin_operator
turn_to satellite3 star3 star1
0
1
0 5 8 10
1
end_operator
begin_operator
turn_to satellite3 star3 star11
0
1
0 5 9 10
1
end_operator
begin_operator
turn_to satellite3 star3 star4
0
1
0 5 11 10
1
end_operator
begin_operator
turn_to satellite3 star3 star5
0
1
0 5 12 10
1
end_operator
begin_operator
turn_to satellite3 star3 star8
0
1
0 5 13 10
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation0
0
1
0 5 0 11
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation2
0
1
0 5 1 11
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation6
0
1
0 5 2 11
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation7
0
1
0 5 3 11
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation9
0
1
0 5 4 11
1
end_operator
begin_operator
turn_to satellite3 star4 phenomenon12
0
1
0 5 5 11
1
end_operator
begin_operator
turn_to satellite3 star4 phenomenon13
0
1
0 5 6 11
1
end_operator
begin_operator
turn_to satellite3 star4 planet10
0
1
0 5 7 11
1
end_operator
begin_operator
turn_to satellite3 star4 star1
0
1
0 5 8 11
1
end_operator
begin_operator
turn_to satellite3 star4 star11
0
1
0 5 9 11
1
end_operator
begin_operator
turn_to satellite3 star4 star3
0
1
0 5 10 11
1
end_operator
begin_operator
turn_to satellite3 star4 star5
0
1
0 5 12 11
1
end_operator
begin_operator
turn_to satellite3 star4 star8
0
1
0 5 13 11
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation0
0
1
0 5 0 12
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation2
0
1
0 5 1 12
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation6
0
1
0 5 2 12
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation7
0
1
0 5 3 12
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation9
0
1
0 5 4 12
1
end_operator
begin_operator
turn_to satellite3 star5 phenomenon12
0
1
0 5 5 12
1
end_operator
begin_operator
turn_to satellite3 star5 phenomenon13
0
1
0 5 6 12
1
end_operator
begin_operator
turn_to satellite3 star5 planet10
0
1
0 5 7 12
1
end_operator
begin_operator
turn_to satellite3 star5 star1
0
1
0 5 8 12
1
end_operator
begin_operator
turn_to satellite3 star5 star11
0
1
0 5 9 12
1
end_operator
begin_operator
turn_to satellite3 star5 star3
0
1
0 5 10 12
1
end_operator
begin_operator
turn_to satellite3 star5 star4
0
1
0 5 11 12
1
end_operator
begin_operator
turn_to satellite3 star5 star8
0
1
0 5 13 12
1
end_operator
begin_operator
turn_to satellite3 star8 groundstation0
0
1
0 5 0 13
1
end_operator
begin_operator
turn_to satellite3 star8 groundstation2
0
1
0 5 1 13
1
end_operator
begin_operator
turn_to satellite3 star8 groundstation6
0
1
0 5 2 13
1
end_operator
begin_operator
turn_to satellite3 star8 groundstation7
0
1
0 5 3 13
1
end_operator
begin_operator
turn_to satellite3 star8 groundstation9
0
1
0 5 4 13
1
end_operator
begin_operator
turn_to satellite3 star8 phenomenon12
0
1
0 5 5 13
1
end_operator
begin_operator
turn_to satellite3 star8 phenomenon13
0
1
0 5 6 13
1
end_operator
begin_operator
turn_to satellite3 star8 planet10
0
1
0 5 7 13
1
end_operator
begin_operator
turn_to satellite3 star8 star1
0
1
0 5 8 13
1
end_operator
begin_operator
turn_to satellite3 star8 star11
0
1
0 5 9 13
1
end_operator
begin_operator
turn_to satellite3 star8 star3
0
1
0 5 10 13
1
end_operator
begin_operator
turn_to satellite3 star8 star4
0
1
0 5 11 13
1
end_operator
begin_operator
turn_to satellite3 star8 star5
0
1
0 5 12 13
1
end_operator
0
