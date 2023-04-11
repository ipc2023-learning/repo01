(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	instrument4 - instrument
	satellite3 - satellite
	instrument5 - instrument
	satellite4 - satellite
	instrument6 - instrument
	instrument7 - instrument
	image2 - mode
	thermograph0 - mode
	image1 - mode
	Star0 - direction
	GroundStation1 - direction
	Star9 - direction
	GroundStation3 - direction
	GroundStation6 - direction
	Star7 - direction
	Star5 - direction
	Star4 - direction
	GroundStation2 - direction
	GroundStation8 - direction
	Planet10 - direction
	Star11 - direction
	Star12 - direction
	Star13 - direction
	Planet14 - direction
	Star15 - direction
	Star16 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 image2)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 GroundStation2)
	(calibration_target instrument0 GroundStation3)
	(supports instrument1 image2)
	(supports instrument1 image1)
	(calibration_target instrument1 GroundStation6)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 GroundStation3)
	(supports instrument2 thermograph0)
	(supports instrument2 image1)
	(supports instrument2 image2)
	(calibration_target instrument2 GroundStation6)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 GroundStation6)
	(supports instrument3 thermograph0)
	(supports instrument3 image1)
	(calibration_target instrument3 Star5)
	(calibration_target instrument3 Star7)
	(supports instrument4 thermograph0)
	(supports instrument4 image1)
	(supports instrument4 image2)
	(calibration_target instrument4 Star5)
	(on_board instrument3 satellite2)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star16)
	(supports instrument5 image1)
	(supports instrument5 image2)
	(calibration_target instrument5 Star4)
	(calibration_target instrument5 GroundStation2)
	(on_board instrument5 satellite3)
	(power_avail satellite3)
	(pointing satellite3 Star11)
	(supports instrument6 image2)
	(supports instrument6 image1)
	(supports instrument6 thermograph0)
	(calibration_target instrument6 GroundStation8)
	(calibration_target instrument6 GroundStation2)
	(calibration_target instrument6 Star4)
	(supports instrument7 image2)
	(supports instrument7 image1)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 GroundStation8)
	(on_board instrument6 satellite4)
	(on_board instrument7 satellite4)
	(power_avail satellite4)
	(pointing satellite4 GroundStation1)
)
(:goal (and
	(pointing satellite2 Star13)
	(have_image Planet10 image1)
	(have_image Star11 image2)
	(have_image Star12 image2)
	(have_image Star13 image1)
	(have_image Planet14 thermograph0)
	(have_image Star15 thermograph0)
	(have_image Star16 image1)
))

)
