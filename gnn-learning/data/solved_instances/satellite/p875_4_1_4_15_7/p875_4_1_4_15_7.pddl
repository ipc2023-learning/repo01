(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	satellite3 - satellite
	instrument3 - instrument
	thermograph2 - mode
	image1 - mode
	thermograph0 - mode
	thermograph3 - mode
	GroundStation4 - direction
	GroundStation5 - direction
	Star9 - direction
	Star10 - direction
	GroundStation14 - direction
	Star0 - direction
	Star1 - direction
	Star2 - direction
	Star11 - direction
	GroundStation3 - direction
	GroundStation8 - direction
	GroundStation12 - direction
	Star6 - direction
	GroundStation7 - direction
	Star13 - direction
	Phenomenon15 - direction
	Phenomenon16 - direction
	Phenomenon17 - direction
	Planet18 - direction
	Planet19 - direction
	Planet20 - direction
	Star21 - direction
)
(:init
	(supports instrument0 thermograph2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 Star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Star21)
	(supports instrument1 thermograph2)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 Star2)
	(calibration_target instrument1 Star1)
	(calibration_target instrument1 Star6)
	(calibration_target instrument1 GroundStation8)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Planet19)
	(supports instrument2 image1)
	(calibration_target instrument2 GroundStation3)
	(calibration_target instrument2 Star13)
	(calibration_target instrument2 Star11)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star6)
	(supports instrument3 thermograph3)
	(supports instrument3 image1)
	(calibration_target instrument3 Star13)
	(calibration_target instrument3 GroundStation7)
	(calibration_target instrument3 Star6)
	(calibration_target instrument3 GroundStation12)
	(calibration_target instrument3 GroundStation8)
	(on_board instrument3 satellite3)
	(power_avail satellite3)
	(pointing satellite3 Star1)
)
(:goal (and
	(have_image Phenomenon15 thermograph2)
	(have_image Phenomenon16 thermograph3)
	(have_image Phenomenon17 thermograph0)
	(have_image Planet18 thermograph0)
	(have_image Planet19 thermograph3)
	(have_image Planet20 thermograph3)
	(have_image Star21 thermograph0)
))

)
