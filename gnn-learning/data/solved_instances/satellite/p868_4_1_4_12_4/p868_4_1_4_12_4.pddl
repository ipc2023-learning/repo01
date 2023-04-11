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
	thermograph3 - mode
	image1 - mode
	thermograph0 - mode
	thermograph2 - mode
	Star1 - direction
	GroundStation4 - direction
	Star6 - direction
	GroundStation3 - direction
	Star9 - direction
	Star10 - direction
	GroundStation7 - direction
	Star11 - direction
	GroundStation8 - direction
	Star0 - direction
	GroundStation5 - direction
	Star2 - direction
	Planet12 - direction
	Planet13 - direction
	Planet14 - direction
	Planet15 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image1)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 Star10)
	(calibration_target instrument0 Star9)
	(calibration_target instrument0 GroundStation3)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Planet12)
	(supports instrument1 thermograph2)
	(calibration_target instrument1 Star10)
	(calibration_target instrument1 Star9)
	(calibration_target instrument1 GroundStation7)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 GroundStation7)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 Star0)
	(calibration_target instrument2 GroundStation8)
	(calibration_target instrument2 Star11)
	(calibration_target instrument2 GroundStation7)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star9)
	(supports instrument3 image1)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 Star2)
	(calibration_target instrument3 GroundStation5)
	(on_board instrument3 satellite3)
	(power_avail satellite3)
	(pointing satellite3 GroundStation4)
)
(:goal (and
	(pointing satellite1 GroundStation5)
	(have_image Planet12 thermograph0)
	(have_image Planet13 thermograph3)
	(have_image Planet14 image1)
	(have_image Planet15 thermograph3)
))

)
