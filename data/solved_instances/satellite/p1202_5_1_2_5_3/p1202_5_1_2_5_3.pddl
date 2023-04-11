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
	GroundStation3 - direction
	GroundStation4 - direction
	GroundStation1 - direction
	GroundStation2 - direction
	Star0 - direction
	Star5 - direction
	Star6 - direction
	Phenomenon7 - direction
)
(:init
	(supports instrument0 image1)
	(calibration_target instrument0 GroundStation1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Star5)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 GroundStation2)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Star6)
	(supports instrument2 image1)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 GroundStation2)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 GroundStation3)
	(supports instrument3 image1)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 GroundStation2)
	(on_board instrument3 satellite3)
	(power_avail satellite3)
	(pointing satellite3 Star0)
	(supports instrument4 image1)
	(supports instrument4 thermograph0)
	(calibration_target instrument4 Star0)
	(on_board instrument4 satellite4)
	(power_avail satellite4)
	(pointing satellite4 Star0)
)
(:goal (and
	(pointing satellite3 GroundStation4)
	(pointing satellite4 GroundStation2)
	(have_image Star5 thermograph0)
	(have_image Star6 image1)
	(have_image Phenomenon7 thermograph0)
))

)
