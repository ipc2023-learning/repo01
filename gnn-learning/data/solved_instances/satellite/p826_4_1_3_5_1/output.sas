begin_version
3
end_version
begin_metric
0
end_metric
34
begin_variable
var0
-1
2
Atom calibrated(instrument0)
NegatedAtom calibrated(instrument0)
end_variable
begin_variable
var1
-1
2
Atom calibrated(instrument1)
NegatedAtom calibrated(instrument1)
end_variable
begin_variable
var2
-1
2
Atom calibrated(instrument2)
NegatedAtom calibrated(instrument2)
end_variable
begin_variable
var3
-1
2
Atom calibrated(instrument3)
NegatedAtom calibrated(instrument3)
end_variable
begin_variable
var4
-1
2
Atom have_image(groundstation0, image1)
NegatedAtom have_image(groundstation0, image1)
end_variable
begin_variable
var5
-1
2
Atom have_image(groundstation0, thermograph0)
NegatedAtom have_image(groundstation0, thermograph0)
end_variable
begin_variable
var6
-1
2
Atom have_image(groundstation0, thermograph2)
NegatedAtom have_image(groundstation0, thermograph2)
end_variable
begin_variable
var7
-1
2
Atom have_image(groundstation1, image1)
NegatedAtom have_image(groundstation1, image1)
end_variable
begin_variable
var8
-1
2
Atom have_image(groundstation1, thermograph0)
NegatedAtom have_image(groundstation1, thermograph0)
end_variable
begin_variable
var9
-1
2
Atom have_image(groundstation1, thermograph2)
NegatedAtom have_image(groundstation1, thermograph2)
end_variable
begin_variable
var10
-1
2
Atom have_image(groundstation3, image1)
NegatedAtom have_image(groundstation3, image1)
end_variable
begin_variable
var11
-1
2
Atom have_image(groundstation3, thermograph0)
NegatedAtom have_image(groundstation3, thermograph0)
end_variable
begin_variable
var12
-1
2
Atom have_image(groundstation3, thermograph2)
NegatedAtom have_image(groundstation3, thermograph2)
end_variable
begin_variable
var13
-1
2
Atom have_image(groundstation4, image1)
NegatedAtom have_image(groundstation4, image1)
end_variable
begin_variable
var14
-1
2
Atom have_image(groundstation4, thermograph0)
NegatedAtom have_image(groundstation4, thermograph0)
end_variable
begin_variable
var15
-1
2
Atom have_image(groundstation4, thermograph2)
NegatedAtom have_image(groundstation4, thermograph2)
end_variable
begin_variable
var16
-1
2
Atom have_image(star2, image1)
NegatedAtom have_image(star2, image1)
end_variable
begin_variable
var17
-1
2
Atom have_image(star2, thermograph0)
NegatedAtom have_image(star2, thermograph0)
end_variable
begin_variable
var18
-1
2
Atom have_image(star2, thermograph2)
NegatedAtom have_image(star2, thermograph2)
end_variable
begin_variable
var19
-1
2
Atom have_image(star5, image1)
NegatedAtom have_image(star5, image1)
end_variable
begin_variable
var20
-1
2
Atom have_image(star5, thermograph0)
NegatedAtom have_image(star5, thermograph0)
end_variable
begin_variable
var21
-1
2
Atom have_image(star5, thermograph2)
NegatedAtom have_image(star5, thermograph2)
end_variable
begin_variable
var22
-1
6
Atom pointing(satellite0, groundstation0)
Atom pointing(satellite0, groundstation1)
Atom pointing(satellite0, groundstation3)
Atom pointing(satellite0, groundstation4)
Atom pointing(satellite0, star2)
Atom pointing(satellite0, star5)
end_variable
begin_variable
var23
-1
6
Atom pointing(satellite1, groundstation0)
Atom pointing(satellite1, groundstation1)
Atom pointing(satellite1, groundstation3)
Atom pointing(satellite1, groundstation4)
Atom pointing(satellite1, star2)
Atom pointing(satellite1, star5)
end_variable
begin_variable
var24
-1
6
Atom pointing(satellite2, groundstation0)
Atom pointing(satellite2, groundstation1)
Atom pointing(satellite2, groundstation3)
Atom pointing(satellite2, groundstation4)
Atom pointing(satellite2, star2)
Atom pointing(satellite2, star5)
end_variable
begin_variable
var25
-1
6
Atom pointing(satellite3, groundstation0)
Atom pointing(satellite3, groundstation1)
Atom pointing(satellite3, groundstation3)
Atom pointing(satellite3, groundstation4)
Atom pointing(satellite3, star2)
Atom pointing(satellite3, star5)
end_variable
begin_variable
var26
-1
2
Atom power_avail(satellite0)
NegatedAtom power_avail(satellite0)
end_variable
begin_variable
var27
-1
2
Atom power_avail(satellite1)
NegatedAtom power_avail(satellite1)
end_variable
begin_variable
var28
-1
2
Atom power_avail(satellite2)
NegatedAtom power_avail(satellite2)
end_variable
begin_variable
var29
-1
2
Atom power_avail(satellite3)
NegatedAtom power_avail(satellite3)
end_variable
begin_variable
var30
-1
2
Atom power_on(instrument0)
NegatedAtom power_on(instrument0)
end_variable
begin_variable
var31
-1
2
Atom power_on(instrument1)
NegatedAtom power_on(instrument1)
end_variable
begin_variable
var32
-1
2
Atom power_on(instrument2)
NegatedAtom power_on(instrument2)
end_variable
begin_variable
var33
-1
2
Atom power_on(instrument3)
NegatedAtom power_on(instrument3)
end_variable
4
begin_mutex_group
6
22 0
22 1
22 2
22 3
22 4
22 5
end_mutex_group
begin_mutex_group
6
23 0
23 1
23 2
23 3
23 4
23 5
end_mutex_group
begin_mutex_group
6
24 0
24 1
24 2
24 3
24 4
24 5
end_mutex_group
begin_mutex_group
6
25 0
25 1
25 2
25 3
25 4
25 5
end_mutex_group
begin_state
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
3
3
2
5
0
0
0
0
1
1
1
1
end_state
begin_goal
2
19 0
24 2
end_goal
180
begin_operator
calibrate satellite0 instrument0 groundstation3
2
22 2
30 0
1
0 0 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument1 groundstation1
2
23 1
31 0
1
0 1 -1 0
1
end_operator
begin_operator
calibrate satellite2 instrument2 groundstation3
2
24 2
32 0
1
0 2 -1 0
1
end_operator
begin_operator
calibrate satellite3 instrument3 star2
2
25 4
33 0
1
0 3 -1 0
1
end_operator
begin_operator
switch_off instrument0 satellite0
0
2
0 26 -1 0
0 30 0 1
1
end_operator
begin_operator
switch_off instrument1 satellite1
0
2
0 27 -1 0
0 31 0 1
1
end_operator
begin_operator
switch_off instrument2 satellite2
0
2
0 28 -1 0
0 32 0 1
1
end_operator
begin_operator
switch_off instrument3 satellite3
0
2
0 29 -1 0
0 33 0 1
1
end_operator
begin_operator
switch_on instrument0 satellite0
0
3
0 0 -1 1
0 26 0 1
0 30 -1 0
1
end_operator
begin_operator
switch_on instrument1 satellite1
0
3
0 1 -1 1
0 27 0 1
0 31 -1 0
1
end_operator
begin_operator
switch_on instrument2 satellite2
0
3
0 2 -1 1
0 28 0 1
0 32 -1 0
1
end_operator
begin_operator
switch_on instrument3 satellite3
0
3
0 3 -1 1
0 29 0 1
0 33 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation0 instrument0 thermograph0
3
0 0
22 0
30 0
1
0 5 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation0 instrument0 thermograph2
3
0 0
22 0
30 0
1
0 6 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation1 instrument0 thermograph0
3
0 0
22 1
30 0
1
0 8 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation1 instrument0 thermograph2
3
0 0
22 1
30 0
1
0 9 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation3 instrument0 thermograph0
3
0 0
22 2
30 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation3 instrument0 thermograph2
3
0 0
22 2
30 0
1
0 12 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation4 instrument0 thermograph0
3
0 0
22 3
30 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation4 instrument0 thermograph2
3
0 0
22 3
30 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite0 star2 instrument0 thermograph0
3
0 0
22 4
30 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite0 star2 instrument0 thermograph2
3
0 0
22 4
30 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite0 star5 instrument0 thermograph0
3
0 0
22 5
30 0
1
0 20 -1 0
1
end_operator
begin_operator
take_image satellite0 star5 instrument0 thermograph2
3
0 0
22 5
30 0
1
0 21 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation0 instrument1 image1
3
1 0
23 0
31 0
1
0 4 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation0 instrument1 thermograph0
3
1 0
23 0
31 0
1
0 5 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation1 instrument1 image1
3
1 0
23 1
31 0
1
0 7 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation1 instrument1 thermograph0
3
1 0
23 1
31 0
1
0 8 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation3 instrument1 image1
3
1 0
23 2
31 0
1
0 10 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation3 instrument1 thermograph0
3
1 0
23 2
31 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation4 instrument1 image1
3
1 0
23 3
31 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation4 instrument1 thermograph0
3
1 0
23 3
31 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite1 star2 instrument1 image1
3
1 0
23 4
31 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite1 star2 instrument1 thermograph0
3
1 0
23 4
31 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite1 star5 instrument1 image1
3
1 0
23 5
31 0
1
0 19 -1 0
1
end_operator
begin_operator
take_image satellite1 star5 instrument1 thermograph0
3
1 0
23 5
31 0
1
0 20 -1 0
1
end_operator
begin_operator
take_image satellite2 groundstation0 instrument2 image1
3
2 0
24 0
32 0
1
0 4 -1 0
1
end_operator
begin_operator
take_image satellite2 groundstation0 instrument2 thermograph0
3
2 0
24 0
32 0
1
0 5 -1 0
1
end_operator
begin_operator
take_image satellite2 groundstation0 instrument2 thermograph2
3
2 0
24 0
32 0
1
0 6 -1 0
1
end_operator
begin_operator
take_image satellite2 groundstation1 instrument2 image1
3
2 0
24 1
32 0
1
0 7 -1 0
1
end_operator
begin_operator
take_image satellite2 groundstation1 instrument2 thermograph0
3
2 0
24 1
32 0
1
0 8 -1 0
1
end_operator
begin_operator
take_image satellite2 groundstation1 instrument2 thermograph2
3
2 0
24 1
32 0
1
0 9 -1 0
1
end_operator
begin_operator
take_image satellite2 groundstation3 instrument2 image1
3
2 0
24 2
32 0
1
0 10 -1 0
1
end_operator
begin_operator
take_image satellite2 groundstation3 instrument2 thermograph0
3
2 0
24 2
32 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite2 groundstation3 instrument2 thermograph2
3
2 0
24 2
32 0
1
0 12 -1 0
1
end_operator
begin_operator
take_image satellite2 groundstation4 instrument2 image1
3
2 0
24 3
32 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite2 groundstation4 instrument2 thermograph0
3
2 0
24 3
32 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite2 groundstation4 instrument2 thermograph2
3
2 0
24 3
32 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite2 star2 instrument2 image1
3
2 0
24 4
32 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite2 star2 instrument2 thermograph0
3
2 0
24 4
32 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite2 star2 instrument2 thermograph2
3
2 0
24 4
32 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite2 star5 instrument2 image1
3
2 0
24 5
32 0
1
0 19 -1 0
1
end_operator
begin_operator
take_image satellite2 star5 instrument2 thermograph0
3
2 0
24 5
32 0
1
0 20 -1 0
1
end_operator
begin_operator
take_image satellite2 star5 instrument2 thermograph2
3
2 0
24 5
32 0
1
0 21 -1 0
1
end_operator
begin_operator
take_image satellite3 groundstation0 instrument3 thermograph0
3
3 0
25 0
33 0
1
0 5 -1 0
1
end_operator
begin_operator
take_image satellite3 groundstation1 instrument3 thermograph0
3
3 0
25 1
33 0
1
0 8 -1 0
1
end_operator
begin_operator
take_image satellite3 groundstation3 instrument3 thermograph0
3
3 0
25 2
33 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite3 groundstation4 instrument3 thermograph0
3
3 0
25 3
33 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite3 star2 instrument3 thermograph0
3
3 0
25 4
33 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite3 star5 instrument3 thermograph0
3
3 0
25 5
33 0
1
0 20 -1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation1
0
1
0 22 1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation3
0
1
0 22 2 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation4
0
1
0 22 3 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star2
0
1
0 22 4 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star5
0
1
0 22 5 0
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation0
0
1
0 22 0 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation3
0
1
0 22 2 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 groundstation4
0
1
0 22 3 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star2
0
1
0 22 4 1
1
end_operator
begin_operator
turn_to satellite0 groundstation1 star5
0
1
0 22 5 1
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation0
0
1
0 22 0 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation1
0
1
0 22 1 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation4
0
1
0 22 3 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star2
0
1
0 22 4 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star5
0
1
0 22 5 2
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation0
0
1
0 22 0 3
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation1
0
1
0 22 1 3
1
end_operator
begin_operator
turn_to satellite0 groundstation4 groundstation3
0
1
0 22 2 3
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star2
0
1
0 22 4 3
1
end_operator
begin_operator
turn_to satellite0 groundstation4 star5
0
1
0 22 5 3
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation0
0
1
0 22 0 4
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation1
0
1
0 22 1 4
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation3
0
1
0 22 2 4
1
end_operator
begin_operator
turn_to satellite0 star2 groundstation4
0
1
0 22 3 4
1
end_operator
begin_operator
turn_to satellite0 star2 star5
0
1
0 22 5 4
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation0
0
1
0 22 0 5
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation1
0
1
0 22 1 5
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation3
0
1
0 22 2 5
1
end_operator
begin_operator
turn_to satellite0 star5 groundstation4
0
1
0 22 3 5
1
end_operator
begin_operator
turn_to satellite0 star5 star2
0
1
0 22 4 5
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation1
0
1
0 23 1 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation3
0
1
0 23 2 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation4
0
1
0 23 3 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star2
0
1
0 23 4 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star5
0
1
0 23 5 0
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation0
0
1
0 23 0 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation3
0
1
0 23 2 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 groundstation4
0
1
0 23 3 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star2
0
1
0 23 4 1
1
end_operator
begin_operator
turn_to satellite1 groundstation1 star5
0
1
0 23 5 1
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation0
0
1
0 23 0 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation1
0
1
0 23 1 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation4
0
1
0 23 3 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star2
0
1
0 23 4 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star5
0
1
0 23 5 2
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation0
0
1
0 23 0 3
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation1
0
1
0 23 1 3
1
end_operator
begin_operator
turn_to satellite1 groundstation4 groundstation3
0
1
0 23 2 3
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star2
0
1
0 23 4 3
1
end_operator
begin_operator
turn_to satellite1 groundstation4 star5
0
1
0 23 5 3
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation0
0
1
0 23 0 4
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation1
0
1
0 23 1 4
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation3
0
1
0 23 2 4
1
end_operator
begin_operator
turn_to satellite1 star2 groundstation4
0
1
0 23 3 4
1
end_operator
begin_operator
turn_to satellite1 star2 star5
0
1
0 23 5 4
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation0
0
1
0 23 0 5
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation1
0
1
0 23 1 5
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation3
0
1
0 23 2 5
1
end_operator
begin_operator
turn_to satellite1 star5 groundstation4
0
1
0 23 3 5
1
end_operator
begin_operator
turn_to satellite1 star5 star2
0
1
0 23 4 5
1
end_operator
begin_operator
turn_to satellite2 groundstation0 groundstation1
0
1
0 24 1 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 groundstation3
0
1
0 24 2 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 groundstation4
0
1
0 24 3 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 star2
0
1
0 24 4 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 star5
0
1
0 24 5 0
1
end_operator
begin_operator
turn_to satellite2 groundstation1 groundstation0
0
1
0 24 0 1
1
end_operator
begin_operator
turn_to satellite2 groundstation1 groundstation3
0
1
0 24 2 1
1
end_operator
begin_operator
turn_to satellite2 groundstation1 groundstation4
0
1
0 24 3 1
1
end_operator
begin_operator
turn_to satellite2 groundstation1 star2
0
1
0 24 4 1
1
end_operator
begin_operator
turn_to satellite2 groundstation1 star5
0
1
0 24 5 1
1
end_operator
begin_operator
turn_to satellite2 groundstation3 groundstation0
0
1
0 24 0 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 groundstation1
0
1
0 24 1 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 groundstation4
0
1
0 24 3 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star2
0
1
0 24 4 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star5
0
1
0 24 5 2
1
end_operator
begin_operator
turn_to satellite2 groundstation4 groundstation0
0
1
0 24 0 3
1
end_operator
begin_operator
turn_to satellite2 groundstation4 groundstation1
0
1
0 24 1 3
1
end_operator
begin_operator
turn_to satellite2 groundstation4 groundstation3
0
1
0 24 2 3
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star2
0
1
0 24 4 3
1
end_operator
begin_operator
turn_to satellite2 groundstation4 star5
0
1
0 24 5 3
1
end_operator
begin_operator
turn_to satellite2 star2 groundstation0
0
1
0 24 0 4
1
end_operator
begin_operator
turn_to satellite2 star2 groundstation1
0
1
0 24 1 4
1
end_operator
begin_operator
turn_to satellite2 star2 groundstation3
0
1
0 24 2 4
1
end_operator
begin_operator
turn_to satellite2 star2 groundstation4
0
1
0 24 3 4
1
end_operator
begin_operator
turn_to satellite2 star2 star5
0
1
0 24 5 4
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation0
0
1
0 24 0 5
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation1
0
1
0 24 1 5
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation3
0
1
0 24 2 5
1
end_operator
begin_operator
turn_to satellite2 star5 groundstation4
0
1
0 24 3 5
1
end_operator
begin_operator
turn_to satellite2 star5 star2
0
1
0 24 4 5
1
end_operator
begin_operator
turn_to satellite3 groundstation0 groundstation1
0
1
0 25 1 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 groundstation3
0
1
0 25 2 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 groundstation4
0
1
0 25 3 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 star2
0
1
0 25 4 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 star5
0
1
0 25 5 0
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation0
0
1
0 25 0 1
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation3
0
1
0 25 2 1
1
end_operator
begin_operator
turn_to satellite3 groundstation1 groundstation4
0
1
0 25 3 1
1
end_operator
begin_operator
turn_to satellite3 groundstation1 star2
0
1
0 25 4 1
1
end_operator
begin_operator
turn_to satellite3 groundstation1 star5
0
1
0 25 5 1
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation0
0
1
0 25 0 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation1
0
1
0 25 1 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation4
0
1
0 25 3 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 star2
0
1
0 25 4 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 star5
0
1
0 25 5 2
1
end_operator
begin_operator
turn_to satellite3 groundstation4 groundstation0
0
1
0 25 0 3
1
end_operator
begin_operator
turn_to satellite3 groundstation4 groundstation1
0
1
0 25 1 3
1
end_operator
begin_operator
turn_to satellite3 groundstation4 groundstation3
0
1
0 25 2 3
1
end_operator
begin_operator
turn_to satellite3 groundstation4 star2
0
1
0 25 4 3
1
end_operator
begin_operator
turn_to satellite3 groundstation4 star5
0
1
0 25 5 3
1
end_operator
begin_operator
turn_to satellite3 star2 groundstation0
0
1
0 25 0 4
1
end_operator
begin_operator
turn_to satellite3 star2 groundstation1
0
1
0 25 1 4
1
end_operator
begin_operator
turn_to satellite3 star2 groundstation3
0
1
0 25 2 4
1
end_operator
begin_operator
turn_to satellite3 star2 groundstation4
0
1
0 25 3 4
1
end_operator
begin_operator
turn_to satellite3 star2 star5
0
1
0 25 5 4
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation0
0
1
0 25 0 5
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation1
0
1
0 25 1 5
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation3
0
1
0 25 2 5
1
end_operator
begin_operator
turn_to satellite3 star5 groundstation4
0
1
0 25 3 5
1
end_operator
begin_operator
turn_to satellite3 star5 star2
0
1
0 25 4 5
1
end_operator
0
