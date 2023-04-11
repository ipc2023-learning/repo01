begin_version
3
end_version
begin_metric
0
end_metric
9
begin_variable
var0
-1
2
Atom power_avail(satellite1)
NegatedAtom power_avail(satellite1)
end_variable
begin_variable
var1
-1
2
Atom power_on(instrument1)
NegatedAtom power_on(instrument1)
end_variable
begin_variable
var2
-1
2
Atom power_avail(satellite0)
NegatedAtom power_avail(satellite0)
end_variable
begin_variable
var3
-1
2
Atom power_on(instrument0)
NegatedAtom power_on(instrument0)
end_variable
begin_variable
var4
-1
11
Atom pointing(satellite1, star0)
Atom pointing(satellite1, star1)
Atom pointing(satellite1, star10)
Atom pointing(satellite1, star2)
Atom pointing(satellite1, star3)
Atom pointing(satellite1, star4)
Atom pointing(satellite1, star5)
Atom pointing(satellite1, star6)
Atom pointing(satellite1, star7)
Atom pointing(satellite1, star8)
Atom pointing(satellite1, star9)
end_variable
begin_variable
var5
-1
11
Atom pointing(satellite0, star0)
Atom pointing(satellite0, star1)
Atom pointing(satellite0, star10)
Atom pointing(satellite0, star2)
Atom pointing(satellite0, star3)
Atom pointing(satellite0, star4)
Atom pointing(satellite0, star5)
Atom pointing(satellite0, star6)
Atom pointing(satellite0, star7)
Atom pointing(satellite0, star8)
Atom pointing(satellite0, star9)
end_variable
begin_variable
var6
-1
2
Atom calibrated(instrument1)
NegatedAtom calibrated(instrument1)
end_variable
begin_variable
var7
-1
2
Atom calibrated(instrument0)
NegatedAtom calibrated(instrument0)
end_variable
begin_variable
var8
-1
2
Atom have_image(star10, image1)
NegatedAtom have_image(star10, image1)
end_variable
0
begin_state
0
1
0
1
4
4
1
1
1
end_state
begin_goal
1
8 0
end_goal
230
begin_operator
calibrate satellite0 instrument0 star3
2
5 4
3 0
1
0 7 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument1 star0
2
4 0
1 0
1
0 6 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument1 star1
2
4 1
1 0
1
0 6 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument1 star5
2
4 6
1 0
1
0 6 -1 0
1
end_operator
begin_operator
switch_off instrument0 satellite0
0
2
0 2 -1 0
0 3 0 1
1
end_operator
begin_operator
switch_off instrument1 satellite1
0
2
0 0 -1 0
0 1 0 1
1
end_operator
begin_operator
switch_on instrument0 satellite0
0
3
0 7 -1 1
0 2 0 1
0 3 -1 0
1
end_operator
begin_operator
switch_on instrument1 satellite1
0
3
0 6 -1 1
0 0 0 1
0 1 -1 0
1
end_operator
begin_operator
take_image satellite0 star10 instrument0 image1
3
7 0
5 2
3 0
1
0 8 -1 0
1
end_operator
begin_operator
take_image satellite1 star10 instrument1 image1
3
6 0
4 2
1 0
1
0 8 -1 0
1
end_operator
begin_operator
turn_to satellite0 star0 star1
0
1
0 5 1 0
1
end_operator
begin_operator
turn_to satellite0 star0 star10
0
1
0 5 2 0
1
end_operator
begin_operator
turn_to satellite0 star0 star2
0
1
0 5 3 0
1
end_operator
begin_operator
turn_to satellite0 star0 star3
0
1
0 5 4 0
1
end_operator
begin_operator
turn_to satellite0 star0 star4
0
1
0 5 5 0
1
end_operator
begin_operator
turn_to satellite0 star0 star5
0
1
0 5 6 0
1
end_operator
begin_operator
turn_to satellite0 star0 star6
0
1
0 5 7 0
1
end_operator
begin_operator
turn_to satellite0 star0 star7
0
1
0 5 8 0
1
end_operator
begin_operator
turn_to satellite0 star0 star8
0
1
0 5 9 0
1
end_operator
begin_operator
turn_to satellite0 star0 star9
0
1
0 5 10 0
1
end_operator
begin_operator
turn_to satellite0 star1 star0
0
1
0 5 0 1
1
end_operator
begin_operator
turn_to satellite0 star1 star10
0
1
0 5 2 1
1
end_operator
begin_operator
turn_to satellite0 star1 star2
0
1
0 5 3 1
1
end_operator
begin_operator
turn_to satellite0 star1 star3
0
1
0 5 4 1
1
end_operator
begin_operator
turn_to satellite0 star1 star4
0
1
0 5 5 1
1
end_operator
begin_operator
turn_to satellite0 star1 star5
0
1
0 5 6 1
1
end_operator
begin_operator
turn_to satellite0 star1 star6
0
1
0 5 7 1
1
end_operator
begin_operator
turn_to satellite0 star1 star7
0
1
0 5 8 1
1
end_operator
begin_operator
turn_to satellite0 star1 star8
0
1
0 5 9 1
1
end_operator
begin_operator
turn_to satellite0 star1 star9
0
1
0 5 10 1
1
end_operator
begin_operator
turn_to satellite0 star10 star0
0
1
0 5 0 2
1
end_operator
begin_operator
turn_to satellite0 star10 star1
0
1
0 5 1 2
1
end_operator
begin_operator
turn_to satellite0 star10 star2
0
1
0 5 3 2
1
end_operator
begin_operator
turn_to satellite0 star10 star3
0
1
0 5 4 2
1
end_operator
begin_operator
turn_to satellite0 star10 star4
0
1
0 5 5 2
1
end_operator
begin_operator
turn_to satellite0 star10 star5
0
1
0 5 6 2
1
end_operator
begin_operator
turn_to satellite0 star10 star6
0
1
0 5 7 2
1
end_operator
begin_operator
turn_to satellite0 star10 star7
0
1
0 5 8 2
1
end_operator
begin_operator
turn_to satellite0 star10 star8
0
1
0 5 9 2
1
end_operator
begin_operator
turn_to satellite0 star10 star9
0
1
0 5 10 2
1
end_operator
begin_operator
turn_to satellite0 star2 star0
0
1
0 5 0 3
1
end_operator
begin_operator
turn_to satellite0 star2 star1
0
1
0 5 1 3
1
end_operator
begin_operator
turn_to satellite0 star2 star10
0
1
0 5 2 3
1
end_operator
begin_operator
turn_to satellite0 star2 star3
0
1
0 5 4 3
1
end_operator
begin_operator
turn_to satellite0 star2 star4
0
1
0 5 5 3
1
end_operator
begin_operator
turn_to satellite0 star2 star5
0
1
0 5 6 3
1
end_operator
begin_operator
turn_to satellite0 star2 star6
0
1
0 5 7 3
1
end_operator
begin_operator
turn_to satellite0 star2 star7
0
1
0 5 8 3
1
end_operator
begin_operator
turn_to satellite0 star2 star8
0
1
0 5 9 3
1
end_operator
begin_operator
turn_to satellite0 star2 star9
0
1
0 5 10 3
1
end_operator
begin_operator
turn_to satellite0 star3 star0
0
1
0 5 0 4
1
end_operator
begin_operator
turn_to satellite0 star3 star1
0
1
0 5 1 4
1
end_operator
begin_operator
turn_to satellite0 star3 star10
0
1
0 5 2 4
1
end_operator
begin_operator
turn_to satellite0 star3 star2
0
1
0 5 3 4
1
end_operator
begin_operator
turn_to satellite0 star3 star4
0
1
0 5 5 4
1
end_operator
begin_operator
turn_to satellite0 star3 star5
0
1
0 5 6 4
1
end_operator
begin_operator
turn_to satellite0 star3 star6
0
1
0 5 7 4
1
end_operator
begin_operator
turn_to satellite0 star3 star7
0
1
0 5 8 4
1
end_operator
begin_operator
turn_to satellite0 star3 star8
0
1
0 5 9 4
1
end_operator
begin_operator
turn_to satellite0 star3 star9
0
1
0 5 10 4
1
end_operator
begin_operator
turn_to satellite0 star4 star0
0
1
0 5 0 5
1
end_operator
begin_operator
turn_to satellite0 star4 star1
0
1
0 5 1 5
1
end_operator
begin_operator
turn_to satellite0 star4 star10
0
1
0 5 2 5
1
end_operator
begin_operator
turn_to satellite0 star4 star2
0
1
0 5 3 5
1
end_operator
begin_operator
turn_to satellite0 star4 star3
0
1
0 5 4 5
1
end_operator
begin_operator
turn_to satellite0 star4 star5
0
1
0 5 6 5
1
end_operator
begin_operator
turn_to satellite0 star4 star6
0
1
0 5 7 5
1
end_operator
begin_operator
turn_to satellite0 star4 star7
0
1
0 5 8 5
1
end_operator
begin_operator
turn_to satellite0 star4 star8
0
1
0 5 9 5
1
end_operator
begin_operator
turn_to satellite0 star4 star9
0
1
0 5 10 5
1
end_operator
begin_operator
turn_to satellite0 star5 star0
0
1
0 5 0 6
1
end_operator
begin_operator
turn_to satellite0 star5 star1
0
1
0 5 1 6
1
end_operator
begin_operator
turn_to satellite0 star5 star10
0
1
0 5 2 6
1
end_operator
begin_operator
turn_to satellite0 star5 star2
0
1
0 5 3 6
1
end_operator
begin_operator
turn_to satellite0 star5 star3
0
1
0 5 4 6
1
end_operator
begin_operator
turn_to satellite0 star5 star4
0
1
0 5 5 6
1
end_operator
begin_operator
turn_to satellite0 star5 star6
0
1
0 5 7 6
1
end_operator
begin_operator
turn_to satellite0 star5 star7
0
1
0 5 8 6
1
end_operator
begin_operator
turn_to satellite0 star5 star8
0
1
0 5 9 6
1
end_operator
begin_operator
turn_to satellite0 star5 star9
0
1
0 5 10 6
1
end_operator
begin_operator
turn_to satellite0 star6 star0
0
1
0 5 0 7
1
end_operator
begin_operator
turn_to satellite0 star6 star1
0
1
0 5 1 7
1
end_operator
begin_operator
turn_to satellite0 star6 star10
0
1
0 5 2 7
1
end_operator
begin_operator
turn_to satellite0 star6 star2
0
1
0 5 3 7
1
end_operator
begin_operator
turn_to satellite0 star6 star3
0
1
0 5 4 7
1
end_operator
begin_operator
turn_to satellite0 star6 star4
0
1
0 5 5 7
1
end_operator
begin_operator
turn_to satellite0 star6 star5
0
1
0 5 6 7
1
end_operator
begin_operator
turn_to satellite0 star6 star7
0
1
0 5 8 7
1
end_operator
begin_operator
turn_to satellite0 star6 star8
0
1
0 5 9 7
1
end_operator
begin_operator
turn_to satellite0 star6 star9
0
1
0 5 10 7
1
end_operator
begin_operator
turn_to satellite0 star7 star0
0
1
0 5 0 8
1
end_operator
begin_operator
turn_to satellite0 star7 star1
0
1
0 5 1 8
1
end_operator
begin_operator
turn_to satellite0 star7 star10
0
1
0 5 2 8
1
end_operator
begin_operator
turn_to satellite0 star7 star2
0
1
0 5 3 8
1
end_operator
begin_operator
turn_to satellite0 star7 star3
0
1
0 5 4 8
1
end_operator
begin_operator
turn_to satellite0 star7 star4
0
1
0 5 5 8
1
end_operator
begin_operator
turn_to satellite0 star7 star5
0
1
0 5 6 8
1
end_operator
begin_operator
turn_to satellite0 star7 star6
0
1
0 5 7 8
1
end_operator
begin_operator
turn_to satellite0 star7 star8
0
1
0 5 9 8
1
end_operator
begin_operator
turn_to satellite0 star7 star9
0
1
0 5 10 8
1
end_operator
begin_operator
turn_to satellite0 star8 star0
0
1
0 5 0 9
1
end_operator
begin_operator
turn_to satellite0 star8 star1
0
1
0 5 1 9
1
end_operator
begin_operator
turn_to satellite0 star8 star10
0
1
0 5 2 9
1
end_operator
begin_operator
turn_to satellite0 star8 star2
0
1
0 5 3 9
1
end_operator
begin_operator
turn_to satellite0 star8 star3
0
1
0 5 4 9
1
end_operator
begin_operator
turn_to satellite0 star8 star4
0
1
0 5 5 9
1
end_operator
begin_operator
turn_to satellite0 star8 star5
0
1
0 5 6 9
1
end_operator
begin_operator
turn_to satellite0 star8 star6
0
1
0 5 7 9
1
end_operator
begin_operator
turn_to satellite0 star8 star7
0
1
0 5 8 9
1
end_operator
begin_operator
turn_to satellite0 star8 star9
0
1
0 5 10 9
1
end_operator
begin_operator
turn_to satellite0 star9 star0
0
1
0 5 0 10
1
end_operator
begin_operator
turn_to satellite0 star9 star1
0
1
0 5 1 10
1
end_operator
begin_operator
turn_to satellite0 star9 star10
0
1
0 5 2 10
1
end_operator
begin_operator
turn_to satellite0 star9 star2
0
1
0 5 3 10
1
end_operator
begin_operator
turn_to satellite0 star9 star3
0
1
0 5 4 10
1
end_operator
begin_operator
turn_to satellite0 star9 star4
0
1
0 5 5 10
1
end_operator
begin_operator
turn_to satellite0 star9 star5
0
1
0 5 6 10
1
end_operator
begin_operator
turn_to satellite0 star9 star6
0
1
0 5 7 10
1
end_operator
begin_operator
turn_to satellite0 star9 star7
0
1
0 5 8 10
1
end_operator
begin_operator
turn_to satellite0 star9 star8
0
1
0 5 9 10
1
end_operator
begin_operator
turn_to satellite1 star0 star1
0
1
0 4 1 0
1
end_operator
begin_operator
turn_to satellite1 star0 star10
0
1
0 4 2 0
1
end_operator
begin_operator
turn_to satellite1 star0 star2
0
1
0 4 3 0
1
end_operator
begin_operator
turn_to satellite1 star0 star3
0
1
0 4 4 0
1
end_operator
begin_operator
turn_to satellite1 star0 star4
0
1
0 4 5 0
1
end_operator
begin_operator
turn_to satellite1 star0 star5
0
1
0 4 6 0
1
end_operator
begin_operator
turn_to satellite1 star0 star6
0
1
0 4 7 0
1
end_operator
begin_operator
turn_to satellite1 star0 star7
0
1
0 4 8 0
1
end_operator
begin_operator
turn_to satellite1 star0 star8
0
1
0 4 9 0
1
end_operator
begin_operator
turn_to satellite1 star0 star9
0
1
0 4 10 0
1
end_operator
begin_operator
turn_to satellite1 star1 star0
0
1
0 4 0 1
1
end_operator
begin_operator
turn_to satellite1 star1 star10
0
1
0 4 2 1
1
end_operator
begin_operator
turn_to satellite1 star1 star2
0
1
0 4 3 1
1
end_operator
begin_operator
turn_to satellite1 star1 star3
0
1
0 4 4 1
1
end_operator
begin_operator
turn_to satellite1 star1 star4
0
1
0 4 5 1
1
end_operator
begin_operator
turn_to satellite1 star1 star5
0
1
0 4 6 1
1
end_operator
begin_operator
turn_to satellite1 star1 star6
0
1
0 4 7 1
1
end_operator
begin_operator
turn_to satellite1 star1 star7
0
1
0 4 8 1
1
end_operator
begin_operator
turn_to satellite1 star1 star8
0
1
0 4 9 1
1
end_operator
begin_operator
turn_to satellite1 star1 star9
0
1
0 4 10 1
1
end_operator
begin_operator
turn_to satellite1 star10 star0
0
1
0 4 0 2
1
end_operator
begin_operator
turn_to satellite1 star10 star1
0
1
0 4 1 2
1
end_operator
begin_operator
turn_to satellite1 star10 star2
0
1
0 4 3 2
1
end_operator
begin_operator
turn_to satellite1 star10 star3
0
1
0 4 4 2
1
end_operator
begin_operator
turn_to satellite1 star10 star4
0
1
0 4 5 2
1
end_operator
begin_operator
turn_to satellite1 star10 star5
0
1
0 4 6 2
1
end_operator
begin_operator
turn_to satellite1 star10 star6
0
1
0 4 7 2
1
end_operator
begin_operator
turn_to satellite1 star10 star7
0
1
0 4 8 2
1
end_operator
begin_operator
turn_to satellite1 star10 star8
0
1
0 4 9 2
1
end_operator
begin_operator
turn_to satellite1 star10 star9
0
1
0 4 10 2
1
end_operator
begin_operator
turn_to satellite1 star2 star0
0
1
0 4 0 3
1
end_operator
begin_operator
turn_to satellite1 star2 star1
0
1
0 4 1 3
1
end_operator
begin_operator
turn_to satellite1 star2 star10
0
1
0 4 2 3
1
end_operator
begin_operator
turn_to satellite1 star2 star3
0
1
0 4 4 3
1
end_operator
begin_operator
turn_to satellite1 star2 star4
0
1
0 4 5 3
1
end_operator
begin_operator
turn_to satellite1 star2 star5
0
1
0 4 6 3
1
end_operator
begin_operator
turn_to satellite1 star2 star6
0
1
0 4 7 3
1
end_operator
begin_operator
turn_to satellite1 star2 star7
0
1
0 4 8 3
1
end_operator
begin_operator
turn_to satellite1 star2 star8
0
1
0 4 9 3
1
end_operator
begin_operator
turn_to satellite1 star2 star9
0
1
0 4 10 3
1
end_operator
begin_operator
turn_to satellite1 star3 star0
0
1
0 4 0 4
1
end_operator
begin_operator
turn_to satellite1 star3 star1
0
1
0 4 1 4
1
end_operator
begin_operator
turn_to satellite1 star3 star10
0
1
0 4 2 4
1
end_operator
begin_operator
turn_to satellite1 star3 star2
0
1
0 4 3 4
1
end_operator
begin_operator
turn_to satellite1 star3 star4
0
1
0 4 5 4
1
end_operator
begin_operator
turn_to satellite1 star3 star5
0
1
0 4 6 4
1
end_operator
begin_operator
turn_to satellite1 star3 star6
0
1
0 4 7 4
1
end_operator
begin_operator
turn_to satellite1 star3 star7
0
1
0 4 8 4
1
end_operator
begin_operator
turn_to satellite1 star3 star8
0
1
0 4 9 4
1
end_operator
begin_operator
turn_to satellite1 star3 star9
0
1
0 4 10 4
1
end_operator
begin_operator
turn_to satellite1 star4 star0
0
1
0 4 0 5
1
end_operator
begin_operator
turn_to satellite1 star4 star1
0
1
0 4 1 5
1
end_operator
begin_operator
turn_to satellite1 star4 star10
0
1
0 4 2 5
1
end_operator
begin_operator
turn_to satellite1 star4 star2
0
1
0 4 3 5
1
end_operator
begin_operator
turn_to satellite1 star4 star3
0
1
0 4 4 5
1
end_operator
begin_operator
turn_to satellite1 star4 star5
0
1
0 4 6 5
1
end_operator
begin_operator
turn_to satellite1 star4 star6
0
1
0 4 7 5
1
end_operator
begin_operator
turn_to satellite1 star4 star7
0
1
0 4 8 5
1
end_operator
begin_operator
turn_to satellite1 star4 star8
0
1
0 4 9 5
1
end_operator
begin_operator
turn_to satellite1 star4 star9
0
1
0 4 10 5
1
end_operator
begin_operator
turn_to satellite1 star5 star0
0
1
0 4 0 6
1
end_operator
begin_operator
turn_to satellite1 star5 star1
0
1
0 4 1 6
1
end_operator
begin_operator
turn_to satellite1 star5 star10
0
1
0 4 2 6
1
end_operator
begin_operator
turn_to satellite1 star5 star2
0
1
0 4 3 6
1
end_operator
begin_operator
turn_to satellite1 star5 star3
0
1
0 4 4 6
1
end_operator
begin_operator
turn_to satellite1 star5 star4
0
1
0 4 5 6
1
end_operator
begin_operator
turn_to satellite1 star5 star6
0
1
0 4 7 6
1
end_operator
begin_operator
turn_to satellite1 star5 star7
0
1
0 4 8 6
1
end_operator
begin_operator
turn_to satellite1 star5 star8
0
1
0 4 9 6
1
end_operator
begin_operator
turn_to satellite1 star5 star9
0
1
0 4 10 6
1
end_operator
begin_operator
turn_to satellite1 star6 star0
0
1
0 4 0 7
1
end_operator
begin_operator
turn_to satellite1 star6 star1
0
1
0 4 1 7
1
end_operator
begin_operator
turn_to satellite1 star6 star10
0
1
0 4 2 7
1
end_operator
begin_operator
turn_to satellite1 star6 star2
0
1
0 4 3 7
1
end_operator
begin_operator
turn_to satellite1 star6 star3
0
1
0 4 4 7
1
end_operator
begin_operator
turn_to satellite1 star6 star4
0
1
0 4 5 7
1
end_operator
begin_operator
turn_to satellite1 star6 star5
0
1
0 4 6 7
1
end_operator
begin_operator
turn_to satellite1 star6 star7
0
1
0 4 8 7
1
end_operator
begin_operator
turn_to satellite1 star6 star8
0
1
0 4 9 7
1
end_operator
begin_operator
turn_to satellite1 star6 star9
0
1
0 4 10 7
1
end_operator
begin_operator
turn_to satellite1 star7 star0
0
1
0 4 0 8
1
end_operator
begin_operator
turn_to satellite1 star7 star1
0
1
0 4 1 8
1
end_operator
begin_operator
turn_to satellite1 star7 star10
0
1
0 4 2 8
1
end_operator
begin_operator
turn_to satellite1 star7 star2
0
1
0 4 3 8
1
end_operator
begin_operator
turn_to satellite1 star7 star3
0
1
0 4 4 8
1
end_operator
begin_operator
turn_to satellite1 star7 star4
0
1
0 4 5 8
1
end_operator
begin_operator
turn_to satellite1 star7 star5
0
1
0 4 6 8
1
end_operator
begin_operator
turn_to satellite1 star7 star6
0
1
0 4 7 8
1
end_operator
begin_operator
turn_to satellite1 star7 star8
0
1
0 4 9 8
1
end_operator
begin_operator
turn_to satellite1 star7 star9
0
1
0 4 10 8
1
end_operator
begin_operator
turn_to satellite1 star8 star0
0
1
0 4 0 9
1
end_operator
begin_operator
turn_to satellite1 star8 star1
0
1
0 4 1 9
1
end_operator
begin_operator
turn_to satellite1 star8 star10
0
1
0 4 2 9
1
end_operator
begin_operator
turn_to satellite1 star8 star2
0
1
0 4 3 9
1
end_operator
begin_operator
turn_to satellite1 star8 star3
0
1
0 4 4 9
1
end_operator
begin_operator
turn_to satellite1 star8 star4
0
1
0 4 5 9
1
end_operator
begin_operator
turn_to satellite1 star8 star5
0
1
0 4 6 9
1
end_operator
begin_operator
turn_to satellite1 star8 star6
0
1
0 4 7 9
1
end_operator
begin_operator
turn_to satellite1 star8 star7
0
1
0 4 8 9
1
end_operator
begin_operator
turn_to satellite1 star8 star9
0
1
0 4 10 9
1
end_operator
begin_operator
turn_to satellite1 star9 star0
0
1
0 4 0 10
1
end_operator
begin_operator
turn_to satellite1 star9 star1
0
1
0 4 1 10
1
end_operator
begin_operator
turn_to satellite1 star9 star10
0
1
0 4 2 10
1
end_operator
begin_operator
turn_to satellite1 star9 star2
0
1
0 4 3 10
1
end_operator
begin_operator
turn_to satellite1 star9 star3
0
1
0 4 4 10
1
end_operator
begin_operator
turn_to satellite1 star9 star4
0
1
0 4 5 10
1
end_operator
begin_operator
turn_to satellite1 star9 star5
0
1
0 4 6 10
1
end_operator
begin_operator
turn_to satellite1 star9 star6
0
1
0 4 7 10
1
end_operator
begin_operator
turn_to satellite1 star9 star7
0
1
0 4 8 10
1
end_operator
begin_operator
turn_to satellite1 star9 star8
0
1
0 4 9 10
1
end_operator
0
