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
	satellite4 - satellite
	instrument4 - instrument
	image1 - mode
	thermograph0 - mode
	GroundStation2 - direction
	GroundStation4 - direction
	Star5 - direction
	GroundStation1 - direction
	GroundStation6 - direction
	Star0 - direction
	GroundStation3 - direction
	Star7 - direction
	Phenomenon8 - direction
	Phenomenon9 - direction
	Star10 - direction
	Star11 - direction
	Planet12 - direction
	Phenomenon13 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 GroundStation1)
	(calibration_target instrument0 GroundStation6)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 GroundStation6)
	(supports instrument1 thermograph0)
	(supports instrument1 image1)
	(calibration_target instrument1 GroundStation3)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Planet12)
	(supports instrument2 image1)
	(calibration_target instrument2 GroundStation1)
	(calibration_target instrument2 GroundStation3)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star5)
	(supports instrument3 thermograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 Star0)
	(calibration_target instrument3 GroundStation6)
	(on_board instrument3 satellite3)
	(power_avail satellite3)
	(pointing satellite3 Star0)
	(supports instrument4 image1)
	(calibration_target instrument4 GroundStation3)
	(on_board instrument4 satellite4)
	(power_avail satellite4)
	(pointing satellite4 GroundStation3)
)
(:goal (and
	(pointing satellite2 GroundStation6)
	(have_image Star7 thermograph0)
	(have_image Phenomenon8 image1)
	(have_image Phenomenon9 image1)
	(have_image Star10 image1)
	(have_image Star11 thermograph0)
	(have_image Planet12 image1)
	(have_image Phenomenon13 image1)
))

)
