(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	image1 - mode
	spectrograph0 - mode
	image3 - mode
	infrared2 - mode
	spectrograph4 - mode
	GroundStation3 - direction
	Star4 - direction
	GroundStation0 - direction
	Star1 - direction
	Star2 - direction
	Phenomenon5 - direction
	Phenomenon6 - direction
	Star7 - direction
	Phenomenon8 - direction
	Phenomenon9 - direction
	Star10 - direction
	Planet11 - direction
)
(:init
	(supports instrument0 image3)
	(supports instrument0 spectrograph0)
	(supports instrument0 infrared2)
	(calibration_target instrument0 GroundStation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 GroundStation3)
	(supports instrument1 image3)
	(supports instrument1 infrared2)
	(supports instrument1 spectrograph4)
	(calibration_target instrument1 Star1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Star10)
	(supports instrument2 image1)
	(supports instrument2 image3)
	(supports instrument2 spectrograph4)
	(calibration_target instrument2 Star2)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Phenomenon6)
)
(:goal (and
	(have_image Phenomenon5 infrared2)
	(have_image Phenomenon6 image3)
	(have_image Star7 spectrograph4)
	(have_image Phenomenon8 spectrograph4)
	(have_image Phenomenon9 image3)
	(have_image Star10 spectrograph4)
	(have_image Planet11 spectrograph4)
))

)
