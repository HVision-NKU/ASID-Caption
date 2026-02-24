PROMPT_LIST = [
    "Provide a comprehensive description of all the content in the video, leaving out no details, and naturally covering the scene, characters, objects, actions, narrative elements, speech, camera, and emotions in a single coherent account.",
    "Thoroughly describe everything in the video, capturing every detail across the scene, characters, objects, actions, narrative elements, speech, camera, and emotions in a clear and unified description.",
    "Please describe all the information in the video without sparing any detail, seamlessly incorporating the scene, characters, objects, actions, narrative elements, speech, camera, and emotions in one coherent paragraph.",
    "Offer a detailed description of the video that covers every aspect of its content, including the scene, characters, objects, actions, narrative elements, speech, camera, and emotions, presented as a single coherent paragraph.",
    "Describe every aspect of the video in full detail, covering the scene, characters, objects, actions, narrative elements, speech, camera, and emotions in a unified and coherent manner.",
    "Please provide a thorough description of all the content in the video, naturally addressing the scene, characters, objects, actions, narrative elements, speech, camera, and emotions in one coherent paragraph.",
    "Give a detailed account of everything in the video, capturing all specifics related to the scene, characters, objects, actions, narrative elements, speech, camera, and emotions in a single, coherent description.",
]

SINGLE_ATTR_PROMPT_TEMPLATES = [
    "Describe the {attr} in the video in detail. Write your answer as one coherent paragraph.",
    "Provide a comprehensive description of the video's {attr}. Write your answer as one coherent paragraph.",
    "Thoroughly describe the {attr} shown in the video. Write your answer as one coherent paragraph.",
    "Give a detailed account of the {attr} in the video. Write your answer as one coherent paragraph.",
    "Please describe the {attr} of the video with rich detail. Write your answer as one coherent paragraph.",
]

PROMPT_LIST = [
    "Provide a comprehensive description of all the content in the video, leaving out no details. Be sure to include as much of the audio information as possible, and ensure that your descriptions of the audio and video are closely aligned.",
    "Thoroughly describe everything in the video, capturing every detail. Include as much information from the audio as possible, and ensure that the descriptions of both audio and video are well-coordinated.",
    "Please describe all the information in the video without sparing every detail in it. As you describe, you should also describe as much of the information in the audio as possible, and pay attention to the synchronization between the audio and video descriptions.",
    "Offer a detailed description of the video, making sure to include every detail. Also, incorporate as much information from the audio as you can, and ensure that your descriptions of the audio and video are in sync.",
    "Describe every aspect of the video in full detail, covering all the information it contains. Additionally, include as much of the audio content as you can, and make sure your descriptions of the audio and video are synchronized.",
    "Please provide a thorough description of all the content in the video, including every detail. As you describe, ensure that you also cover as much information from the audio as possible, and be mindful of the synchronization between the audio and video as you do so.",
    "Give a detailed account of everything in the video, capturing all the specifics. While doing so, also include as much information from the audio as possible, ensuring that the descriptions of audio and video are well-synchronized."
]

ATTR_KEYS = ["scene", "characters", "objects", "actions", "narrative_elements", "speech", "camera", "emotions"]