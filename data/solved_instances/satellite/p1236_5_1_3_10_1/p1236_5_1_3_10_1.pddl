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
	image2 - mode
	Star0 - direction
	Star4 - direction
	Star5 - direction
	GroundStation6 - direction
	Star7 - direction
	GroundStation1 - direction
	Star9 - direction
	GroundStation8 - direction
	GroundStation2 - direction
	GroundStation3 - direction
	Planet10 - direction
)
(:init
	(supports instrument0 thermograph0)
	(supports instrument0 image2)
	(supports instrument0 image1)
	(calibration_target instrument0 GroundStation3)
	(calibration_target instrument0 Star7)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 GroundStation2)
	(supports instrument1 image2)
	(calibration_target instrument1 GroundStation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Star4)
	(supports instrument2 image1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 GroundStation2)
	(calibration_target instrument2 Star9)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star0)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 GroundStation8)
	(on_board instrument3 satellite3)
	(power_avail satellite3)
	(pointing satellite3 Planet10)
	(supports instrument4 image2)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 GroundStation3)
	(calibration_target instrument4 GroundStation2)
	(on_board instrument4 satellite4)
	(power_avail satellite4)
	(pointing satellite4 GroundStation3)
)
(:goal (and
	(pointing satellite0 Planet10)
	(pointing satellite2 Star0)
	(have_image Planet10 image1)
))

)
