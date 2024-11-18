import axios from 'axios';
import { Form, Card } from 'react-bootstrap';
import { useState } from 'react';
import './App.css';

function App() {
  const [ digits, setDigits ] = useState("");

  const changeFile = (e) => {
    // Implement FileReader API to read the file
    let fReader = new FileReader();
    let uploadSrc = e.target; // Set a variable to hold the actual HTML Element
    // Again, uploadSrc is an HTML element, not an actual src on its own

    let fileType = uploadSrc.files[0].type;
    // Check the file type for validity
    if(fileType.includes("image")) {
        if(uploadSrc.files[0].size > 5000000) {
            return alert("File Size Too Large");
        }

        fReader.readAsDataURL(uploadSrc.files[0]);
    } else {
        return alert("Invalid File Type");
    }

    // Load the file if applicable
    fReader.onload = async (e) => {
        let url;
        try {
            url = e.target.result;
            console.log(url);

            axios.post("http://localhost:5050/prediction", {url: url}).then((res) => {
              setDigits(res.data);
            }).catch((res) => {
              setDigits("An Error Occurred");
            });
        }
        catch(e) {
            return alert("There was an error loading image"); // Errors due to renaming file extension
        }
    };
  };

  return (
    <>
      <Card className="mx-5 mt-5 bg-primary">
        <h1 className="text-light display-1 m-auto text-center mb-2">Drop A File Below!</h1>
        <Form.Group controlId="formFile" className="mb-3 mx-5">
          <Form.Control onChange={changeFile} type="file" />
        </Form.Group>
        <h1 className="display-5 m-auto text-center text-light mb-3">Output:</h1>
        <p className="lead mb-3 text-light m-auto text-center">{digits}</p>
      </Card>
    </>
  )
}

export default App
