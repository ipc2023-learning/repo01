(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	instrument3 - instrument
	satellite2 - satellite
	instrument4 - instrument
	instrument5 - instrument
	spectrograph0 - mode
	infrared2 - mode
	image1 - mode
	Star0 - direction
	Star6 - direction
	GroundStation8 - direction
	Star9 - direction
	GroundStation11 - direction
	Star12 - direction
	Star13 - direction
	GroundStation1 - direction
	Star14 - direction
	GroundStation4 - direction
	Star10 - direction
	Star7 - direction
	Star3 - direction
	GroundStation2 - direction
	Star5 - direction
	Star15 - direction
	Phenomenon16 - direction
	Phenomenon17 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 image1)
	(calibration_target instrument0 Star14)
	(calibration_target instrument0 Star7)
	(supports instrument1 spectrograph0)
	(calibration_target instrument1 GroundStation1)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 GroundStation11)
	(supports instrument2 spectrograph0)
	(supports instrument2 image1)
	(supports instrument2 infrared2)
	(calibration_target instrument2 Star14)
	(calibration_target instrument2 Star7)
	(supports instrument3 image1)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 Star5)
	(calibration_target instrument3 GroundStation4)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Star12)
	(supports instrument4 image1)
	(supports instrument4 spectrograph0)
	(supports instrument4 infrared2)
	(calibration_target instrument4 Star7)
	(calibration_target instrument4 Star10)
	(calibration_target instrument4 GroundStation2)
	(supports instrument5 spectrograph0)
	(calibration_target instrument5 Star5)
	(calibration_target instrument5 GroundStation2)
	(calibration_target instrument5 Star3)
	(on_board instrument4 satellite2)
	(on_board instrument5 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star9)
)
(:goal (and
	(have_image Star15 spectrograph0)
	(have_image Phenomenon16 image1)
	(have_image Phenomenon17 infrared2)
))

)
