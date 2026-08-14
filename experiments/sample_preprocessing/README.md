\# Sample Preprocessing Test



\## Objective



Verify that the CrysCo preprocessing pipeline can convert crystal

structure CIF files and property data into a PyTorch Geometric dataset.



\## Input



\- 5 CIF structures

\- `eh.csv`

\- `ehs.csv`



\## Preprocessing



The CrysCo preprocessing pipeline was used to construct:



\- atomic features

\- edge features

\- bond-angle features

\- dihedral-angle features

\- global structural features



\## Result



Successfully processed:



5 / 5 structures



\## Status



PASS



\## Next Step



Perform a small GPU training test using the processed dataset.

