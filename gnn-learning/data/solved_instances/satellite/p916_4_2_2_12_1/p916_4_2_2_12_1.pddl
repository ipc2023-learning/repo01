(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	satellite3 - satellite
	instrument4 - instrument
	instrument5 - instrument
	image1 - mode
	thermograph0 - mode
	GroundStation0 - direction
	GroundStation3 - direction
	Star6 - direction
	Star1 - direction
	GroundStation10 - direction
	Star4 - direction
	GroundStation7 - direction
	GroundStation2 - direction
	GroundStation5 - direction
	GroundStation11 - direction
	Star9 - direction
	GroundStation8 - direction
	Planet12 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 Star6)
	(calibration_target instrument0 GroundStation10)
	(calibration_target instrument0 Star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Star9)
	(supports instrument1 image1)
	(calibration_target instrument1 Star9)
	(calibration_target instrument1 GroundStation5)
	(calibration_target instrument1 GroundStation10)
	(supports instrument2 thermograph0)
	(supports instrument2 image1)
	(calibration_target instrument2 Star6)
	(calibration_target instrument2 GroundStation8)
	(calibration_target instrument2 Star4)
	(calibration_target instrument2 GroundStation2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 GroundStation10)
	(supports instrument3 thermograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 GroundStation2)
	(calibration_target instrument3 Star4)
	(calibration_target instrument3 GroundStation10)
	(calibration_target instrument3 Star1)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 GroundStation5)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(calibration_target instrument4 GroundStation5)
	(calibration_target instrument4 GroundStation2)
	(calibration_target instrument4 Star9)
	(calibration_target instrument4 GroundStation7)
	(supports instrument5 thermograph0)
	(supports instrument5 image1)
	(calibration_target instrument5 GroundStation8)
	(calibration_target instrument5 Star9)
	(calibration_target instrument5 GroundStation11)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(power_avail satellite3)
	(pointing satellite3 Star1)
)
(:goal (and
	(pointing satellite0 GroundStation11)
	(pointing satellite3 GroundStation3)
	(have_image Planet12 image1)
))

)
