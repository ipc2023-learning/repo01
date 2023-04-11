(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	satellite1 - satellite
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	satellite3 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite4 - satellite
	instrument10 - instrument
	instrument11 - instrument
	instrument12 - instrument
	thermograph0 - mode
	image2 - mode
	image1 - mode
	GroundStation3 - direction
	Star0 - direction
	GroundStation1 - direction
	Star4 - direction
	GroundStation2 - direction
	Star5 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 Star4)
	(supports instrument1 image2)
	(calibration_target instrument1 GroundStation2)
	(supports instrument2 thermograph0)
	(supports instrument2 image2)
	(calibration_target instrument2 GroundStation1)
	(supports instrument3 image2)
	(supports instrument3 image1)
	(calibration_target instrument3 GroundStation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Star0)
	(supports instrument4 image2)
	(supports instrument4 image1)
	(calibration_target instrument4 Star0)
	(supports instrument5 thermograph0)
	(supports instrument5 image1)
	(supports instrument5 image2)
	(calibration_target instrument5 GroundStation3)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Star4)
	(supports instrument6 image2)
	(supports instrument6 image1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 Star0)
	(supports instrument7 image1)
	(calibration_target instrument7 GroundStation3)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 GroundStation3)
	(supports instrument8 image2)
	(supports instrument8 image1)
	(supports instrument8 thermograph0)
	(calibration_target instrument8 Star0)
	(supports instrument9 image1)
	(supports instrument9 thermograph0)
	(supports instrument9 image2)
	(calibration_target instrument9 Star4)
	(on_board instrument8 satellite3)
	(on_board instrument9 satellite3)
	(power_avail satellite3)
	(pointing satellite3 Star5)
	(supports instrument10 image1)
	(calibration_target instrument10 GroundStation1)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 Star4)
	(supports instrument12 image2)
	(calibration_target instrument12 GroundStation2)
	(on_board instrument10 satellite4)
	(on_board instrument11 satellite4)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 GroundStation1)
)
(:goal (and
	(pointing satellite2 Star4)
	(pointing satellite4 Star5)
	(have_image Star5 thermograph0)
))

)
