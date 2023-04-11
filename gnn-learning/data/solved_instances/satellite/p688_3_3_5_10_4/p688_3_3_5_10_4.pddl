(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	satellite1 - satellite
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	satellite2 - satellite
	instrument6 - instrument
	instrument7 - instrument
	image3 - mode
	image1 - mode
	infrared2 - mode
	spectrograph4 - mode
	spectrograph0 - mode
	GroundStation3 - direction
	Star4 - direction
	GroundStation8 - direction
	GroundStation7 - direction
	Star1 - direction
	GroundStation0 - direction
	Star9 - direction
	Star5 - direction
	Star2 - direction
	Star6 - direction
	Star10 - direction
	Star11 - direction
	Phenomenon12 - direction
	Phenomenon13 - direction
)
(:init
	(supports instrument0 infrared2)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 Star5)
	(calibration_target instrument0 Star2)
	(calibration_target instrument0 GroundStation7)
	(supports instrument1 spectrograph0)
	(supports instrument1 image3)
	(calibration_target instrument1 Star9)
	(supports instrument2 spectrograph4)
	(supports instrument2 image1)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 Star9)
	(calibration_target instrument2 GroundStation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 GroundStation8)
	(supports instrument3 infrared2)
	(supports instrument3 spectrograph4)
	(supports instrument3 image3)
	(calibration_target instrument3 Star6)
	(calibration_target instrument3 GroundStation7)
	(supports instrument4 spectrograph0)
	(supports instrument4 image3)
	(supports instrument4 image1)
	(calibration_target instrument4 Star9)
	(calibration_target instrument4 GroundStation0)
	(calibration_target instrument4 Star1)
	(supports instrument5 image3)
	(supports instrument5 infrared2)
	(supports instrument5 spectrograph4)
	(calibration_target instrument5 Star5)
	(calibration_target instrument5 Star9)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Star10)
	(supports instrument6 image1)
	(calibration_target instrument6 Star2)
	(calibration_target instrument6 Star5)
	(calibration_target instrument6 Star9)
	(supports instrument7 image3)
	(supports instrument7 infrared2)
	(calibration_target instrument7 Star6)
	(on_board instrument6 satellite2)
	(on_board instrument7 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star10)
)
(:goal (and
	(pointing satellite1 Phenomenon12)
	(have_image Star10 image3)
	(have_image Star11 infrared2)
	(have_image Phenomenon12 spectrograph4)
	(have_image Phenomenon13 spectrograph4)
))

)
