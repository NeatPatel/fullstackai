import axios from 'axios';
import { Form, Card } from 'react-bootstrap';
import { useState } from 'react';
import './App.css';

function App() {
  const [ digits, setDigits ] = useState("");

  const changeFile = (e) => {
    axios.post("http://localhost:5050/prediction", e.target.value).then((res) => {
      setDigits(res.data);
    }).catch((res) => {
      setDigits("An Error Occurred");
    });
  };

  return (
    <>
      <Card className="mx-5 mt-5 bg-primary">
        <h1 className="text-light display-1 m-auto text-center mb-2">Drop A File Below!</h1>
        <Form.Group controlId="formFile" className="mb-3 mx-5">
          <Form.Control onChange={changeFile} type="file" />
        </Form.Group>
        <h1 className="display-5 m-auto text-center text-light">Output:</h1>
        <p className="mb-5 text-light m-auto text-center">{digits}</p>
      </Card>
    </>
  )
}

export default App
